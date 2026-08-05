//! HTTP MCP server exposing proveKV state and page operations.
#![cfg(feature = "mcp-server")]

use crate::{HybridComponent, HybridStateId, StateStore};
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct AppState {
    store: Arc<Mutex<StateStore>>,
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}
#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

fn ok(id: Option<Value>, result: Value) -> Json<RpcResponse> {
    Json(RpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    })
}
fn err(id: Option<Value>, code: i64, message: impl Into<String>) -> Json<RpcResponse> {
    Json(RpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(json!({"code": code, "message": message.into()})),
    })
}

fn tools() -> Value {
    json!({"tools":[
     {"name":"provekv_get_page","description":"Read one validated proveKV page.","inputSchema":{"type":"object","properties":{"run_id":{"type":"string"},"layer":{"type":"integer"},"kv_type":{"type":"string"}},"required":["run_id","layer","kv_type"]}},
     {"name":"provekv_list_pages","description":"List page references for a state/run.","inputSchema":{"type":"object","properties":{"run_id":{"type":"string"}},"required":["run_id"]}},
     {"name":"provekv_fork","description":"Fork a state with a reason recorded in its component inventory.","inputSchema":{"type":"object","properties":{"parent_state_id":{"type":"string"},"reason":{"type":"string"}},"required":["parent_state_id","reason"]}},
     {"name":"provekv_get_receipt","description":"Return the state metadata/receipt for a run.","inputSchema":{"type":"object","properties":{"run_id":{"type":"string"}},"required":["run_id"]}}
    ]})
}

fn arg<'a>(p: &'a Value, name: &str) -> Result<&'a Value, String> {
    p.get(name)
        .ok_or_else(|| format!("missing parameter {name}"))
}
fn state_id(s: &str) -> Result<HybridStateId, String> {
    HybridStateId::try_from(s.to_string()).map_err(|e| e.to_string())
}

async fn handle(State(app): State<AppState>, Json(req): Json<RpcRequest>) -> impl IntoResponse {
    if req.jsonrpc != "2.0" {
        return (
            StatusCode::BAD_REQUEST,
            err(req.id, -32600, "invalid JSON-RPC version"),
        );
    }
    match req.method.as_str() {
        "initialize" => (
            StatusCode::OK,
            ok(
                req.id,
                json!({"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"provekv","version":env!("CARGO_PKG_VERSION")}}),
            ),
        ),
        "notifications/initialized" => (StatusCode::ACCEPTED, ok(None, Value::Null)),
        "tools/list" => (StatusCode::OK, ok(req.id, tools())),
        "tools/call" => {
            let name = req.params.get("name").and_then(Value::as_str).unwrap_or("");
            let a = req.params.get("arguments").cloned().unwrap_or_default();
            let result = call_tool(&app, name, &a);
            match result {
                Ok(v) => (
                    StatusCode::OK,
                    ok(
                        req.id,
                        json!({"content":[{"type":"text","text":serde_json::to_string(&v).unwrap()}],"structuredContent":v}),
                    ),
                ),
                Err(e) => (StatusCode::OK, err(req.id, -32000, e)),
            }
        }
        _ => (
            StatusCode::OK,
            err(req.id, -32601, format!("method not found: {}", req.method)),
        ),
    }
}

fn call_tool(app: &AppState, name: &str, a: &Value) -> Result<Value, String> {
    let mut store = app
        .store
        .lock()
        .map_err(|_| "store lock poisoned".to_string())?;
    match name {
        "provekү_get_page" | "proվkv_get_page" => {
            let id = state_id(arg(a, "run_id")?.as_str().ok_or("run_id must be string")?)?;
            let layer = arg(a, "layer")?.as_u64().ok_or("layer must be integer")? as usize;
            let kv = arg(a, "kv_type")?
                .as_str()
                .ok_or("kv_type must be string")?;
            let node = store.get(&id).ok_or("state not found")?;
            let page = node
                .manifest
                .page_refs
                .get(layer)
                .ok_or("layer not found")?;
            let (header, payload) = store
                .page_store
                .read_page(&page.digest)
                .map_err(|e| e.to_string())?;
            Ok(
                json!({"run_id":id.as_str(),"layer":layer,"kv_type":kv,"page_ref":page,"header":header,"payload":payload}),
            )
        }
        "provekү_list_pages" | "provekv_list_pages" => {
            let id = state_id(arg(a, "run_id")?.as_str().ok_or("run_id must be string")?)?;
            let node = store.get(&id).ok_or("state not found")?;
            Ok(json!({"run_id":id.as_str(),"pages":node.manifest.page_refs}))
        }
        "provekv_fork" => {
            let parent = state_id(
                arg(a, "parent_state_id")?
                    .as_str()
                    .ok_or("parent_state_id must be string")?,
            )?;
            let reason = arg(a, "reason")?.as_str().ok_or("reason must be string")?;
            let p = store.get(&parent).ok_or("parent state not found")?;
            let mut manifest = p.manifest.clone();
            manifest.parent_lineage.push(parent.clone());
            manifest.component_inventory.push(HybridComponent {
                name: "mcp_fork_reason".into(),
                version: "1".into(),
                digest: blake3::hash(reason.as_bytes()).to_hex().to_string(),
            });
            let child = store.fork(&parent, manifest).map_err(|e| e.to_string())?;
            Ok(json!({"parent_state_id":parent.as_str(),"state_id":child.as_str(),"reason":reason}))
        }
        "provekv_get_receipt" => {
            let id = state_id(arg(a, "run_id")?.as_str().ok_or("run_id must be string")?)?;
            let n = store.get(&id).ok_or("state not found")?;
            Ok(
                json!({"run_id":id.as_str(),"state_id":id.as_str(),"parent_state_id":n.parent_id,"page_count":n.manifest.page_refs.len(),"released":n.released,"manifest":n.manifest}),
            )
        }
        _ => Err(format!("unknown tool: {name}")),
    }
}

/// Start the HTTP MCP endpoint on `0.0.0.0:1739`.
pub async fn run(
    root: impl Into<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let state = AppState {
        store: Arc::new(Mutex::new(StateStore::open(root)?)),
    };
    let app = Router::new().route("/mcp", post(handle)).with_state(state);
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::UNSPECIFIED, 1739)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
