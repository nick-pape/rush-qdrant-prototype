//! Filesystem watcher for incremental re-indexing
//!
//! Watches catalog directories for file changes and triggers
//! incremental re-indexing when modifications are detected.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{Watcher, RecursiveMode, Event, EventKind};

use crate::engine::config::should_skip_path;
use crate::engine::chunker::chunk_file;
use crate::engine::HttpEmbedder;
use crate::engine::QdrantUploader;
use crate::engine::util;
use crate::{CatalogConfig, is_text_file, chrono_timestamp};

/// Statistics from an incremental crawl
pub struct CrawlStats {
    pub new_files: usize,
    pub changed_files: usize,
    pub unchanged_files: usize,
    pub deleted_files: usize,
    pub chunks_embedded: usize,
}

/// Run an incremental crawl for a single catalog
pub fn run_incremental_crawl(
    catalog_name: &str,
    catalog_config: &CatalogConfig,
    embedder: &HttpEmbedder,
    collection: &str,
    qdrant_url: Option<&str>,
    qdrant_api_key: Option<&str>,
    vector_size: usize,
) -> anyhow::Result<CrawlStats> {
    let directory = &catalog_config.path;
    let uploader = QdrantUploader::new(collection, qdrant_url, qdrant_api_key, vector_size)?;

    // Get existing files from Qdrant
    let existing_files = uploader.get_catalog_files(catalog_name)?;

    // Scan directory
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
    let mut all_chunks: Vec<crate::engine::Chunk> = Vec::new();

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
            uploader.delete_file(rel_path, catalog_name)?;
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
                for mut chunk in chunks {
                    chunk.breadcrumb = chunk.breadcrumb.replace(":[fallback-split]", "");
                    all_chunks.push(chunk);
                }
            }
            Err(e) => {
                eprintln!(
                    "[{}] Warning: failed to chunk {}: {}",
                    chrono_timestamp(),
                    file_path,
                    e
                );
            }
        }
    }

    // Delete orphaned files
    let mut deleted_count = 0;
    for (rel_path, _) in existing_files.iter() {
        if !rel_files_set.contains(rel_path) {
            uploader.delete_file(rel_path, catalog_name)?;
            deleted_count += 1;
        }
    }

    // Embed via HTTP API and upload
    let total_chunks = all_chunks.len();
    if total_chunks > 0 {
        let mut file_chunks: HashMap<String, usize> = HashMap::new();
        let mut file_expected: HashMap<String, usize> = HashMap::new();

        for batch in all_chunks.chunks(32) {
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
            let embeddings = embedder.embed_batch(&texts)?;

            let embedded: Vec<(crate::engine::Chunk, Vec<f32>)> = batch
                .iter()
                .cloned()
                .zip(embeddings)
                .collect();
            for upload_batch in embedded.chunks(100) {
                uploader.upload_batch(upload_batch)?;
                for (chunk, _) in upload_batch {
                    let fid = util::display_file_id(chunk.file_id);
                    *file_chunks.entry(fid.clone()).or_insert(0) += 1;
                    file_expected.entry(fid).or_insert(chunk.chunk_count);
                }
            }
        }

        for (fid, count) in &file_chunks {
            if Some(count) == file_expected.get(fid) {
                let _ = uploader.mark_file_complete(fid, catalog_name);
            }
        }
    }

    Ok(CrawlStats {
        new_files: new_count,
        changed_files: changed_count,
        unchanged_files: unchanged_count,
        deleted_files: deleted_count,
        chunks_embedded: total_chunks,
    })
}

/// Start a file watcher for a single catalog
pub fn start_watcher(
    catalog_name: String,
    catalog_config: CatalogConfig,
    embedder: Arc<HttpEmbedder>,
    collection: String,
    qdrant_url: Option<String>,
    qdrant_api_key: Option<String>,
    vector_size: usize,
) {
    let watch_path = catalog_config.path.clone();

    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut file_watcher = match notify::recommended_watcher(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    match event.kind {
                        EventKind::Create(_)
                        | EventKind::Modify(_)
                        | EventKind::Remove(_) => {
                            let _ = tx.send(event);
                        }
                        _ => {}
                    }
                }
            },
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!(
                    "[{}] Failed to create watcher for '{}': {}",
                    chrono_timestamp(),
                    catalog_name,
                    e
                );
                return;
            }
        };

        if let Err(e) =
            file_watcher.watch(std::path::Path::new(&watch_path), RecursiveMode::Recursive)
        {
            eprintln!(
                "[{}] Failed to watch '{}': {}",
                chrono_timestamp(),
                watch_path,
                e
            );
            return;
        }

        eprintln!(
            "[{}] Watching '{}' for changes (catalog: {})",
            chrono_timestamp(),
            watch_path,
            catalog_name
        );

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
                eprintln!(
                    "[{}] Changes detected in '{}', re-indexing...",
                    chrono_timestamp(),
                    catalog_name
                );

                match run_incremental_crawl(
                    &catalog_name,
                    &catalog_config,
                    &embedder,
                    &collection,
                    qdrant_url.as_deref(),
                    qdrant_api_key.as_deref(),
                    vector_size,
                ) {
                    Ok(stats) => {
                        let total_changes =
                            stats.new_files + stats.changed_files + stats.deleted_files;
                        if total_changes > 0 {
                            eprintln!(
                                "[{}] Re-indexed '{}': {} new, {} changed, {} deleted ({} chunks)",
                                chrono_timestamp(),
                                catalog_name,
                                stats.new_files,
                                stats.changed_files,
                                stats.deleted_files,
                                stats.chunks_embedded
                            );
                        } else {
                            eprintln!(
                                "[{}] No indexable changes in '{}'",
                                chrono_timestamp(),
                                catalog_name
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[{}] Re-index failed for '{}': {}",
                            chrono_timestamp(),
                            catalog_name,
                            e
                        );
                    }
                }

                has_pending = false;
            }
        }
    });
}
