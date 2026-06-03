//! rush-qdrant: Local stdio MCP for code search with hosted backend.
//!
//! Watches local files, chunks them via tree-sitter AST, and pushes
//! chunks to a hosted code-index-api for embedding + Qdrant storage.
//! Exposes semantic_search and view_chunks MCP tools over stdio.

mod engine;
mod api_client;
mod stdio;
mod watcher;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use engine::{
    partitioner::{partition_typescript, PartitionConfig, ChunkQualityReport, PartitionDebug},
    SMALL_CHUNK_CHARS,
};

const DEFAULT_API_URL: &str = "https://code-index.mcp.pape.house";

/// Catalog configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct CatalogConfig {
    pub(crate) r#type: String,
    pub(crate) path: String,
}

#[derive(Parser)]
#[command(name = "rush-qdrant", version, about = "Code search MCP with hosted backend")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Watch this directory for changes and serve MCP over stdio (default mode)
    #[arg(long)]
    watch: Option<String>,

    /// Catalog type: "monorepo" or "folder" (auto-detected if omitted)
    #[arg(long, default_value = "auto")]
    r#type: String,
}

#[derive(Subcommand)]
enum Commands {
    /// One-shot crawl: index a directory then exit
    Crawl {
        /// Directory to crawl
        path: String,
        /// Catalog type
        #[arg(long, default_value = "auto")]
        r#type: String,
    },
    /// Dump chunks for a TypeScript file (debugging)
    DumpChunks {
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value = "6000")]
        target_size: usize,
        #[arg(long)]
        visualize: bool,
        #[arg(long)]
        with_fallback: bool,
        #[arg(long)]
        debug: bool,
    },
    /// Audit chunking quality across files (debugging)
    AuditChunks {
        #[arg(long, default_value = "20")]
        count: usize,
        #[arg(long)]
        dir: String,
    },
}

fn get_api_url() -> String {
    std::env::var("CODE_INDEX_URL").unwrap_or_else(|_| DEFAULT_API_URL.to_string())
}

fn get_api_token() -> anyhow::Result<String> {
    if let Ok(token) = std::env::var("CODE_INDEX_TOKEN") {
        return Ok(token);
    }
    let token_path = shellexpand::tilde("~/.config/rush-qdrant/token");
    if let Ok(token) = std::fs::read_to_string(token_path.as_ref()) {
        return Ok(token.trim().to_string());
    }
    Err(anyhow::anyhow!(
        "No API token found. Set CODE_INDEX_TOKEN env var or write token to ~/.config/rush-qdrant/token"
    ))
}

fn current_branch(path: &str) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(path)
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() || s == "HEAD" { None } else { Some(s) }
        })
        .unwrap_or_else(|| "main".to_string())
}

fn detect_catalog_type(path: &str) -> String {
    if std::path::Path::new(path).join("rush.json").exists() {
        "monorepo".to_string()
    } else {
        "folder".to_string()
    }
}

fn catalog_name_from_path(path: &str) -> String {
    // Try git remote origin — extract repo name (no owner prefix, avoids slash)
    if let Ok(output) = std::process::Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(path)
        .output()
    {
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !url.is_empty() {
            if let Some(repo) = url.trim_end_matches(".git").rsplit('/').next() {
                if !repo.is_empty() {
                    return repo.to_string();
                }
            }
        }
    }
    // Fallback: directory basename
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

pub(crate) fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let h = (now / 3600) % 24;
    let m = (now / 60) % 60;
    let s = now % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

pub(crate) fn is_text_file(path: &str) -> bool {
    let extensions = [
        "ts", "tsx", "js", "jsx", "md", "mdx", "json",
        "yaml", "yml", "txt", "rst", "toml", "ini", "conf",
    ];
    let path_lower = path.to_lowercase();
    extensions.iter().any(|ext| path_lower.ends_with(&format!(".{}", ext)))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Crawl { path, r#type }) => {
            let api = api_client::ApiClient::new(&get_api_url(), &get_api_token()?)?;
            let resolved_type = if r#type == "auto" { detect_catalog_type(&path) } else { r#type };
            let abs_path = std::fs::canonicalize(&path)?.to_string_lossy().to_string();
            let catalog_name = catalog_name_from_path(&abs_path);
            let config = CatalogConfig { r#type: resolved_type, path: abs_path.clone() };
            let label = current_branch(&abs_path);

            eprintln!("Crawling '{}' as catalog '{}' (branch: {})...", config.path, catalog_name, label);
            let stats = watcher::run_incremental_crawl(&catalog_name, &config, &api, &label)?;
            eprintln!(
                "Done: {} new, {} changed, {} unchanged, {} deleted ({} chunks ingested)",
                stats.new_files, stats.changed_files, stats.unchanged_files,
                stats.deleted_files, stats.chunks_ingested
            );
        }

        Some(Commands::DumpChunks { file, target_size, visualize, with_fallback, debug }) => {
            run_dump_chunks(&file, target_size, visualize, with_fallback, debug)?;
        }

        Some(Commands::AuditChunks { count, dir }) => {
            run_audit_chunks(count, dir)?;
        }

        None => {
            // Default mode: stdio MCP + optional file watcher
            let api = api_client::ApiClient::new(&get_api_url(), &get_api_token()?)?;
            let api = Arc::new(api);

            let mut catalogs: Vec<String> = Vec::new();

            if let Some(ref watch_path) = cli.watch {
                let abs_path = std::fs::canonicalize(&watch_path)?.to_string_lossy().to_string();
                let resolved_type = if cli.r#type == "auto" { detect_catalog_type(&abs_path) } else { cli.r#type };
                let catalog_name = catalog_name_from_path(&abs_path);
                let config = CatalogConfig { r#type: resolved_type, path: abs_path.clone() };

                let label = current_branch(&abs_path);
                catalogs.push(catalog_name.clone());

                let bg_api = api.clone();
                let bg_name = catalog_name.clone();
                let bg_config = config.clone();
                let bg_label = label.clone();
                std::thread::spawn(move || {
                    eprintln!("Initial index for '{}' (branch: {})...", bg_name, bg_label);
                    match watcher::run_incremental_crawl(&bg_name, &bg_config, &bg_api, &bg_label) {
                        Ok(stats) => {
                            eprintln!(
                                "  {} new, {} changed, {} unchanged, {} deleted ({} chunks)",
                                stats.new_files, stats.changed_files, stats.unchanged_files,
                                stats.deleted_files, stats.chunks_ingested
                            );
                        }
                        Err(e) => eprintln!("  Warning: initial crawl failed: {}", e),
                    }
                    watcher::start_watcher(bg_name, bg_config, bg_api, bg_label);
                });
            }

            let active_label = if let Some(ref wp) = cli.watch {
                let abs = std::fs::canonicalize(wp).unwrap_or_default();
                current_branch(&abs.to_string_lossy())
            } else {
                "main".to_string()
            };
            stdio::run_stdio(&api, &catalogs, &active_label);
        }
    }

    Ok(())
}

fn run_dump_chunks(file: &PathBuf, target_size: usize, visualize: bool, with_fallback: bool, enable_debug: bool) -> anyhow::Result<()> {
    println!("📦 Chunks for: {}", file.display());
    if !with_fallback {
        println!("🔍 Strict mode: AST-only (fallback disabled)");
    }
    println!();

    let source = std::fs::read_to_string(file)?;
    let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or("unknown.ts");
    let file_path = file.to_string_lossy().to_string();
    let package_name = engine::package_lookup::find_package_name(&file_path, "");

    let config = PartitionConfig {
        target_size,
        file_name: file_name.to_string(),
        package_name: package_name.clone(),
        debug: PartitionDebug { enabled: enable_debug },
        allow_fallback: with_fallback,
    };

    let chunks = partition_typescript(&source, &config, &file_path, &package_name);
    let report = ChunkQualityReport::from_chunks(&chunks, source.len());

    if visualize {
        let lines: Vec<&str> = source.lines().collect();
        for (i, chunk) in chunks.iter().enumerate() {
            println!("-- [CHUNK {}] [{} lines] [{} chars] --", i + 1, chunk.end_line - chunk.start_line + 1, chunk.text.len());
            for ln in chunk.start_line..=chunk.end_line {
                if ln > 0 && ln <= lines.len() { println!("{}", lines[ln - 1]); }
            }
            println!();
        }
        println!("=== QUALITY SCORE ===");
        println!("Score: {:.1}%  Chunks: {}  Small (<{}): {}", report.score, chunks.len(), SMALL_CHUNK_CHARS, report.small_chunks);
    } else {
        println!("Total chunks: {}  Target: {} chars\n", chunks.len(), target_size);
        for (i, chunk) in chunks.iter().enumerate() {
            println!("━━━ Chunk {} ━━━  {} chars  Lines {}-{}  {}", i + 1, chunk.text.len(), chunk.start_line, chunk.end_line, chunk.breadcrumb);
            for line in chunk.text.lines().take(5) { println!("  {}", line); }
            if chunk.text.lines().count() > 5 { println!("  ..."); }
            println!();
        }
        println!("Quality: {:.1}%", report.score);
    }
    Ok(())
}

fn run_audit_chunks(count: usize, dir: String) -> anyhow::Result<()> {
    use rand::seq::IndexedRandom;
    println!("📊 Sampling {} TypeScript files from: {}\n", count, dir);

    let ts_files: Vec<PathBuf> = walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().is_some_and(|ext| ext == "ts") && !p.to_string_lossy().contains("node_modules")
        })
        .map(|e| e.path().to_owned())
        .collect();

    if ts_files.is_empty() { return Err(anyhow::anyhow!("No TypeScript files found")); }
    println!("Found {} files", ts_files.len());

    let mut rng = rand::rng();
    let mut results: Vec<_> = ts_files.choose_multiple(&mut rng, count)
        .filter_map(|path| {
            let source = std::fs::read_to_string(path).ok()?;
            let config = PartitionConfig {
                file_name: path.file_name()?.to_string_lossy().to_string(),
                package_name: "n/a".to_string(),
                allow_fallback: false,
                ..Default::default()
            };
            let chunks = partition_typescript(&source, &config, path.to_str().unwrap(), "n/a");
            let report = ChunkQualityReport::from_chunks(&chunks, source.len());
            Some((path.clone(), report))
        })
        .collect();

    results.sort_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap());
    for (i, (path, report)) in results.iter().enumerate() {
        println!("{}. {} {}", i + 1, report.format(), path.strip_prefix(&dir).unwrap_or(path).display());
    }
    Ok(())
}
