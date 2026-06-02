//! HTTP-based embedding via an OpenAI-compatible /v1/embeddings endpoint

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

const MAX_BATCH: usize = 4;

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

pub struct HttpEmbedder {
    client: Client,
    url: String,
    model: String,
    api_key: Option<String>,
    pub dimensions: usize,
}

impl HttpEmbedder {
    pub fn new(base_url: &str, model: &str, dimensions: usize, api_key: Option<&str>) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
        Ok(Self {
            client,
            url,
            model: model.to_string(),
            api_key: api_key.map(|s| s.to_string()),
            dimensions,
        })
    }

    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text])?;
        results
            .pop()
            .ok_or_else(|| anyhow!("Empty response from embedding API"))
    }

    fn embed_single_inner(&self, text: &str) -> Result<Vec<f32>> {
        let body = EmbedRequest {
            model: &self.model,
            input: vec![text],
            encoding_format: "float",
        };
        let mut req = self.client.post(&self.url).json(&body);
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().map_err(|e| anyhow!("Embedding request failed: {}", e))?;
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(anyhow!("{}", &text[..text.len().min(200)]));
        }
        let embed_resp: EmbedResponse = resp.json()?;
        embed_resp
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow!("Empty response"))
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        // Adaptive batching: estimate tokens (~4 chars/token) and keep each
        // HTTP request under a token budget to avoid exceeding the server's
        // context size. Conservative budget leaves room for tokenizer overhead.
        const TOKEN_BUDGET: usize = 14000;
        const CHARS_PER_TOKEN: usize = 2;

        let mut cursor = 0;
        while cursor < texts.len() {
            let mut end = cursor + 1;
            let mut total_tokens = texts[cursor].len() / CHARS_PER_TOKEN;
            while end < texts.len() && end - cursor < MAX_BATCH {
                let next_tokens = texts[end].len() / CHARS_PER_TOKEN;
                if total_tokens + next_tokens > TOKEN_BUDGET {
                    break;
                }
                total_tokens += next_tokens;
                end += 1;
            }
            let chunk = &texts[cursor..end];
            cursor = end;
            let body = EmbedRequest {
                model: &self.model,
                input: chunk.to_vec(),
                encoding_format: "float",
            };

            let mut req = self.client.post(&self.url).json(&body);
            if let Some(ref key) = self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req
                .send()
                .map_err(|e| anyhow!("Embedding request failed: {}", e))?;

            if resp.status() == reqwest::StatusCode::BAD_REQUEST && chunk.len() > 1 {
                // Batch too large — fall back to one-at-a-time for this batch
                for &single in chunk {
                    match self.embed_single_inner(single) {
                        Ok(emb) => all_embeddings.push(emb),
                        Err(e) => {
                            eprintln!(
                                "  ⚠️ Skipping oversized chunk ({} chars): {}",
                                single.len(),
                                e
                            );
                            all_embeddings.push(vec![0.0; self.dimensions]);
                        }
                    }
                }
                continue;
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().unwrap_or_default();
                if text.contains("exceed") && chunk.len() == 1 {
                    eprintln!(
                        "  ⚠️ Skipping oversized chunk ({} chars)",
                        chunk[0].len()
                    );
                    all_embeddings.push(vec![0.0; self.dimensions]);
                    continue;
                }
                return Err(anyhow!(
                    "Embedding API returned HTTP {}: {}",
                    status,
                    &text[..text.len().min(500)]
                ));
            }

            let embed_resp: EmbedResponse = resp
                .json()
                .map_err(|e| anyhow!("Failed to parse embedding response: {}", e))?;

            if embed_resp.data.len() != chunk.len() {
                return Err(anyhow!(
                    "Expected {} embeddings, got {}",
                    chunk.len(),
                    embed_resp.data.len()
                ));
            }

            for datum in embed_resp.data {
                if datum.embedding.len() != self.dimensions {
                    return Err(anyhow!(
                        "Expected {}-dim embedding, got {}",
                        self.dimensions,
                        datum.embedding.len()
                    ));
                }
                all_embeddings.push(datum.embedding);
            }
        }

        Ok(all_embeddings)
    }
}
