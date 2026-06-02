//! Filesystem watcher for incremental re-indexing via hosted API.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use notify::{Watcher, RecursiveMode, Event, EventKind};

use crate::api_client::ApiClient;
use crate::engine::config::should_skip_path;
use crate::engine::chunker::chunk_file;
use crate::{CatalogConfig, is_text_file, chrono_timestamp};

pub struct CrawlStats {
    pub new_files: usize,
    pub changed_files: usize,
    pub unchanged_files: usize,
    pub deleted_files: usize,
    pub chunks_ingested: usize,
}

pub fn run_incremental_crawl(
    catalog_name: &str,
    catalog_config: &CatalogConfig,
    client: &ApiClient,
) -> anyhow::Result<CrawlStats> {
    let directory = &catalog_config.path;

    let existing_files = client.get_catalog_files(catalog_name)?;

    let mut files_to_process: Vec<(String, String)> = Vec::new();
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path().to_string_lossy().to_string();
        if !should_skip_path(&path) && is_text_file(&path) {
            let rel_path = path
                .strip_prefix(directory)
                .unwrap_or(&path)
                .trim_start_matches('/')
                .trim_start_matches('\\')
                .to_string();
            files_to_process.push((path, rel_path));
        }
    }

    let rel_files_set: HashSet<String> = files_to_process.iter().map(|(_, rel)| rel.clone()).collect();

    let mut new_count = 0;
    let mut changed_count = 0;
    let mut unchanged_count = 0;
    let mut total_ingested = 0;

    for (file_path, rel_path) in &files_to_process {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let current_hash = format!("sha256:{:x}", hasher.finalize());

        if let Some(existing_info) = existing_files.get(rel_path) {
            if existing_info.content_hash == current_hash && existing_info.file_complete {
                unchanged_count += 1;
                continue;
            }
            client.delete_file(catalog_name, rel_path)?;
            changed_count += 1;
        } else {
            new_count += 1;
        }

        let package_name = if catalog_config.r#type == "monorepo" {
            crate::engine::package_lookup::find_package_name(file_path, directory)
        } else {
            std::path::Path::new(file_path)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or(catalog_name)
                .to_string()
        };

        match chunk_file(file_path, catalog_name, directory, &package_name, 6000) {
            Ok(chunks) => {
                let chunks: Vec<_> = chunks
                    .into_iter()
                    .map(|mut c| {
                        c.breadcrumb = c.breadcrumb.replace(":[fallback-split]", "");
                        c
                    })
                    .collect();
                match client.ingest(&chunks) {
                    Ok(n) => total_ingested += n,
                    Err(e) => {
                        eprintln!(
                            "[{}] Warning: ingest failed for {}: {}",
                            chrono_timestamp(), file_path, e
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[{}] Warning: failed to chunk {}: {}",
                    chrono_timestamp(), file_path, e
                );
            }
        }
    }

    let mut deleted_count = 0;
    for (rel_path, _) in existing_files.iter() {
        if !rel_files_set.contains(rel_path) {
            client.delete_file(catalog_name, rel_path)?;
            deleted_count += 1;
        }
    }

    Ok(CrawlStats {
        new_files: new_count,
        changed_files: changed_count,
        unchanged_files: unchanged_count,
        deleted_files: deleted_count,
        chunks_ingested: total_ingested,
    })
}

pub fn start_watcher(
    catalog_name: String,
    catalog_config: CatalogConfig,
    client: std::sync::Arc<ApiClient>,
) {
    let watch_path = catalog_config.path.clone();

    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut file_watcher = match notify::recommended_watcher(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                            let _ = tx.send(event);
                        }
                        _ => {}
                    }
                }
            },
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("[{}] Failed to create watcher for '{}': {}", chrono_timestamp(), catalog_name, e);
                return;
            }
        };

        if let Err(e) = file_watcher.watch(std::path::Path::new(&watch_path), RecursiveMode::Recursive) {
            eprintln!("[{}] Failed to watch '{}': {}", chrono_timestamp(), watch_path, e);
            return;
        }

        eprintln!("[{}] Watching '{}' for changes (catalog: {})", chrono_timestamp(), watch_path, catalog_name);
        let _watcher = file_watcher;

        let mut has_pending = false;
        let mut quiet_since = Instant::now();

        loop {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(event) => {
                    let has_relevant = event.paths.iter().any(|p| {
                        let path_str = p.to_string_lossy();
                        !should_skip_path(&path_str) && is_text_file(&path_str)
                    });
                    if has_relevant {
                        has_pending = true;
                        quiet_since = Instant::now();
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }

            if has_pending && quiet_since.elapsed() >= Duration::from_secs(2) {
                eprintln!("[{}] Changes detected in '{}', re-indexing...", chrono_timestamp(), catalog_name);

                match run_incremental_crawl(&catalog_name, &catalog_config, &client) {
                    Ok(stats) => {
                        let total = stats.new_files + stats.changed_files + stats.deleted_files;
                        if total > 0 {
                            eprintln!(
                                "[{}] Re-indexed '{}': {} new, {} changed, {} deleted ({} chunks)",
                                chrono_timestamp(), catalog_name,
                                stats.new_files, stats.changed_files, stats.deleted_files, stats.chunks_ingested
                            );
                        } else {
                            eprintln!("[{}] No indexable changes in '{}'", chrono_timestamp(), catalog_name);
                        }
                    }
                    Err(e) => {
                        eprintln!("[{}] Re-index failed for '{}': {}", chrono_timestamp(), catalog_name, e);
                    }
                }
                has_pending = false;
            }
        }
    });
}
