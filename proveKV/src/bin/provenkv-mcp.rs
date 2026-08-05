#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let root = std::env::var_os("PROVEKV_STORE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("./provenkv-store"));
    provekv::mcp_server::run(root).await
}
