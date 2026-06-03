//! HTTP client for the hosted code-index-api REST service.

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct ApiClient {
    client: Client,
    base_url: String,
    token: String,
}

#[derive(Serialize)]
struct IngestRequest<'a> {
    catalog: &'a str,
    label: &'a str,
    file_id: &'a str,
    relative_path: &'a str,
    content_hash: &'a str,
    chunks: Vec<IngestChunk<'a>>,
}

#[derive(Serialize)]
struct IngestChunk<'a> {
    text: &'a str,
    source_uri: &'a str,
    source_type: &'a str,
    start_line: usize,
    end_line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_name: Option<&'a str>,
    chunk_type: &'a str,
    chunk_kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    breadcrumb: Option<&'a str>,
    chunk_number: usize,
    chunk_count: usize,
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<&'a str>,
}

#[derive(Serialize)]
struct ViewRequest<'a> {
    file_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk_end: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub score: f32,
    pub file_id: Option<String>,
    pub relative_path: Option<String>,
    pub catalog: Option<String>,
    pub breadcrumb: Option<String>,
    pub chunk_number: Option<usize>,
    pub chunk_count: Option<usize>,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub symbol_name: Option<String>,
    pub chunk_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ViewChunk {
    pub file_id: Option<String>,
    pub relative_path: Option<String>,
    pub catalog: Option<String>,
    pub breadcrumb: Option<String>,
    pub chunk_number: Option<usize>,
    pub chunk_count: Option<usize>,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub chunk_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FileSyncInfo {
    pub content_hash: String,
    pub file_complete: bool,
}

impl ApiClient {
    pub fn new(base_url: &str, token: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            token: token.to_string(),
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        self.client
            .request(method, format!("{}{}", self.base_url, path))
            .bearer_auth(&self.token)
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
        catalog: Option<&str>,
        label: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let body = SearchRequest {
            query,
            limit,
            catalog,
            label,
        };
        let resp = self
            .request(reqwest::Method::POST, "/v1/search")
            .json(&body)
            .send()?;
        if !resp.status().is_success() {
            return Err(anyhow!("Search failed: HTTP {}", resp.status()));
        }
        Ok(resp.json()?)
    }

    pub fn view(
        &self,
        file_id: &str,
        catalog: Option<&str>,
        chunk_start: Option<usize>,
        chunk_end: Option<usize>,
    ) -> Result<Vec<ViewChunk>> {
        let body = ViewRequest {
            file_id,
            catalog,
            chunk_start,
            chunk_end,
        };
        let resp = self
            .request(reqwest::Method::POST, "/v1/view")
            .json(&body)
            .send()?;
        if !resp.status().is_success() {
            return Err(anyhow!("View failed: HTTP {}", resp.status()));
        }
        Ok(resp.json()?)
    }

    pub fn ingest(&self, chunks: &[crate::engine::Chunk], label: &str) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let first = &chunks[0];
        let body = IngestRequest {
            catalog: &first.catalog,
            label,
            file_id: &crate::engine::util::display_file_id(first.file_id),
            relative_path: &first.relative_path,
            content_hash: &first.content_hash,
            chunks: chunks
                .iter()
                .map(|c| IngestChunk {
                    text: &c.text,
                    source_uri: &c.source_uri,
                    source_type: &c.source_type,
                    start_line: c.start_line,
                    end_line: c.end_line,
                    symbol_name: c.symbol_name.as_deref(),
                    chunk_type: &c.chunk_type,
                    chunk_kind: &c.chunk_kind,
                    breadcrumb: Some(&c.breadcrumb),
                    chunk_number: c.chunk_number,
                    chunk_count: c.chunk_count,
                })
                .collect(),
        };

        let resp = self
            .request(reqwest::Method::POST, "/v1/ingest")
            .json(&body)
            .send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(anyhow!("Ingest failed: HTTP {} - {}", status, &text[..text.len().min(300)]));
        }

        #[derive(Deserialize)]
        struct IngestResponse {
            ingested: usize,
        }
        let r: IngestResponse = resp.json()?;
        Ok(r.ingested)
    }

    pub fn delete_file(&self, catalog: &str, relative_path: &str) -> Result<()> {
        let path = format!(
            "/v1/files/{}/{}",
            catalog,
            urlencoding::encode(relative_path)
        );
        let resp = self.request(reqwest::Method::DELETE, &path).send()?;
        if !resp.status().is_success() {
            return Err(anyhow!("Delete failed: HTTP {}", resp.status()));
        }
        Ok(())
    }

    pub fn get_catalog_files(&self, catalog: &str, label: Option<&str>) -> Result<HashMap<String, FileSyncInfo>> {
        let mut path = format!("/v1/files/{}", urlencoding::encode(catalog));
        if let Some(l) = label {
            path.push_str(&format!("?label={}", urlencoding::encode(l)));
        }
        let resp = self.request(reqwest::Method::GET, &path).send()?;
        if !resp.status().is_success() {
            return Err(anyhow!("Get files failed: HTTP {}", resp.status()));
        }
        Ok(resp.json()?)
    }
}
