//! Stdio MCP transport — JSON-RPC 2.0 over stdin/stdout.

use std::io::{self, BufRead, Write};
use serde::{Deserialize, Serialize};

use crate::api_client::ApiClient;

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    #[serde(default)]
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }
}

fn error(id: Option<serde_json::Value>, code: i32, message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(JsonRpcError { code, message }),
    }
}

fn handle_initialize(id: Option<serde_json::Value>, catalogs: &[String]) -> JsonRpcResponse {
    let _ = catalogs;
    success(
        id,
        serde_json::json!({
            "protocolVersion": "2025-03-26",
            "serverInfo": { "name": "rush-qdrant", "version": "0.2.0" },
            "capabilities": { "tools": {} }
        }),
    )
}

fn handle_tools_list(id: Option<serde_json::Value>, catalogs: &[String]) -> JsonRpcResponse {
    let catalog_desc = if catalogs.len() == 1 {
        format!("Searches the '{}' catalog.", catalogs[0])
    } else if catalogs.is_empty() {
        "No catalogs configured.".to_string()
    } else {
        format!("Available catalogs: {}. Omit to search all.", catalogs.join(", "))
    };

    success(
        id,
        serde_json::json!({
            "tools": [
                {
                    "name": "semantic_search",
                    "description": format!(
                        "Search indexed code and documentation using semantic similarity. \
                         Returns file IDs, similarity scores, breadcrumbs, and code previews. {}",
                        catalog_desc
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Natural language search query" },
                            "limit": { "type": "integer", "description": "Max results (default 10)", "default": 10 },
                            "catalog": { "type": "string", "description": "Filter to a specific catalog (optional)" }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "view_chunks",
                    "description": "Retrieve full chunk content by file ID. Use IDs from semantic_search results. \
                                    Supports selectors: '700a4ba232fe9ddc' (all), '700a4ba232fe9ddc:3' (chunk 3), \
                                    '700a4ba232fe9ddc:2-3' (range), '700a4ba232fe9ddc:3-end' (to end).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "File IDs with optional chunk selectors"
                            }
                        },
                        "required": ["ids"]
                    }
                }
            ]
        }),
    )
}

fn handle_search(
    id: Option<serde_json::Value>,
    args: &serde_json::Value,
    client: &ApiClient,
    label: &str,
) -> JsonRpcResponse {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let catalog = args.get("catalog").and_then(|v| v.as_str());

    if query.is_empty() {
        return error(id, -32602, "Missing required parameter: query".into());
    }

    // Search current branch, fall back to main if empty
    let mut results = client.search(query, limit, catalog, Some(label)).unwrap_or_default();
    if results.is_empty() && label != "main" {
        results = client.search(query, limit, catalog, Some("main")).unwrap_or_default();
    }

    match Ok::<_, anyhow::Error>(results) {
        Ok(results) => {
            let mut output = String::new();
            for r in &results {
                let breadcrumb = r.breadcrumb.as_deref().unwrap_or("unknown");
                let file_id = r.file_id.as_deref().unwrap_or("?");
                let chunk_num = r.chunk_number.unwrap_or(0);
                output.push_str(&format!(
                    "{}:{}  {:.3}  {}\n",
                    file_id, chunk_num, r.score, breadcrumb
                ));
                if let Some(ref text) = r.text {
                    for line in text.lines().take(3) {
                        output.push_str(&format!("> {}\n", line));
                    }
                }
                output.push('\n');
            }
            if results.is_empty() {
                output.push_str("No results found.\n");
            }
            success(
                id,
                serde_json::json!({ "content": [{ "type": "text", "text": output }] }),
            )
        }
        Err(e) => error(id, -32603, format!("Search failed: {}", e)),
    }
}

fn handle_view(
    id: Option<serde_json::Value>,
    args: &serde_json::Value,
    client: &ApiClient,
) -> JsonRpcResponse {
    let ids: Vec<String> = args
        .get("ids")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    if ids.is_empty() {
        return error(id, -32602, "Missing required parameter: ids".into());
    }

    let mut output = String::new();
    for spec in &ids {
        let (file_id, chunk_start, chunk_end) = parse_view_selector(spec);

        match client.view(&file_id, None, chunk_start, chunk_end) {
            Ok(chunks) => {
                if chunks.is_empty() {
                    output.push_str(&format!("{} ERROR: CHUNK NOT FOUND\n\n", spec));
                    continue;
                }
                for c in &chunks {
                    let breadcrumb = c.breadcrumb.as_deref().unwrap_or("unknown");
                    let cn = c.chunk_number.unwrap_or(0);
                    let cc = c.chunk_count.unwrap_or(0);
                    output.push_str(&format!(
                        "{}:{} ({}/{}) {}\n",
                        file_id, cn, cn, cc, breadcrumb
                    ));
                    if let Some(ref rp) = c.relative_path {
                        if let Some(ref cat) = c.catalog {
                            output.push_str(&format!("Source: {}:{}\n", cat, rp));
                        }
                    }
                    output.push_str(&format!(
                        "Lines: {}-{}\nType: {}\n\n",
                        c.start_line.unwrap_or(0),
                        c.end_line.unwrap_or(0),
                        c.chunk_type.as_deref().unwrap_or("?")
                    ));
                    if let Some(ref text) = c.text {
                        for line in text.lines() {
                            output.push_str(&format!("> {}\n", line));
                        }
                    }
                    output.push('\n');
                }
            }
            Err(e) => {
                output.push_str(&format!("{} ERROR: {}\n\n", spec, e));
            }
        }
    }

    success(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": output }] }),
    )
}

fn parse_view_selector(spec: &str) -> (String, Option<usize>, Option<usize>) {
    if let Some(colon) = spec.find(':') {
        let file_id = spec[..colon].to_string();
        let selector = &spec[colon + 1..];
        if selector.ends_with("-end") {
            let start: usize = selector[..selector.len() - 4].parse().unwrap_or(1);
            (file_id, Some(start), None)
        } else if selector.contains('-') {
            let parts: Vec<&str> = selector.split('-').collect();
            let start: usize = parts[0].parse().unwrap_or(1);
            let end: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(start);
            (file_id, Some(start), Some(end))
        } else {
            let n: usize = selector.parse().unwrap_or(1);
            (file_id, Some(n), Some(n))
        }
    } else {
        (spec.to_string(), None, None)
    }
}

pub fn run_stdio(client: &ApiClient, catalogs: &[String], label: &str) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let resp = error(None, -32700, format!("Parse error: {}", e));
                let _ = writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap());
                let _ = stdout.flush();
                continue;
            }
        };

        let resp = match req.method.as_str() {
            "initialize" => handle_initialize(req.id, catalogs),
            "notifications/initialized" => continue,
            "tools/list" => handle_tools_list(req.id, catalogs),
            "tools/call" => {
                let params = req.params.unwrap_or(serde_json::json!({}));
                let tool = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                match tool {
                    "semantic_search" => handle_search(req.id, &args, client, label),
                    "view_chunks" => handle_view(req.id, &args, client),
                    _ => error(req.id, -32601, format!("Unknown tool: {}", tool)),
                }
            }
            _ => {
                if req.id.is_some() {
                    error(req.id, -32601, format!("Method not found: {}", req.method))
                } else {
                    continue;
                }
            }
        };

        let _ = writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap());
        let _ = stdout.flush();
    }
}
