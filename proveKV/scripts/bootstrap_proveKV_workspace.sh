#!/usr/bin/env bash
set -euo pipefail

# Conservative bootstrap for an empty repository.
# This creates the workspace and crate directories only if they are missing.

mkdir -p crates/quant-codec-core/src crates/proveKV/src

if [ ! -f Cargo.toml ]; then
cat > Cargo.toml <<'EOF'
[workspace]
members = [
  "crates/quant-codec-core",
  "crates/proveKV",
]
resolver = "2"

[workspace.package]
edition = "2021"
license = "MIT OR Apache-2.0"
repository = "https://github.com/recursiveintell/proveKV"
rust-version = "1.78"
EOF
fi

if [ ! -f crates/quant-codec-core/Cargo.toml ]; then
cat > crates/quant-codec-core/Cargo.toml <<'EOF'
[package]
name = "quant-codec-core"
version = "0.1.0-alpha.1"
edition.workspace = true
license.workspace = true
rust-version.workspace = true
description = "Shared codec/profile/shape traits for governed compression experiments."

[dependencies]
serde = { version = "1", features = ["derive"], optional = true }
thiserror = "1"
blake3 = "1"

[features]
default = ["serde"]
serde = ["dep:serde"]
EOF
fi

if [ ! -f crates/proveKV/Cargo.toml ]; then
cat > crates/proveKV/Cargo.toml <<'EOF'
[package]
name = "proveKV"
version = "0.1.0-alpha.1"
edition.workspace = true
license.workspace = true
rust-version.workspace = true
description = "Shared compressed KV-cache pool infrastructure for Rust experiments."

[dependencies]
quant-codec-core = { path = "../quant-codec-core" }
serde = { version = "1", features = ["derive"], optional = true }
thiserror = "1"
blake3 = "1"

[dev-dependencies]
proptest = "1"

[features]
default = ["serde"]
serde = ["dep:serde", "quant-codec-core/serde"]
turbo-quant-adapter = []
fibquant-adapter = []
bench = []
EOF
fi

touch crates/quant-codec-core/src/lib.rs crates/proveKV/src/lib.rs

echo "bootstrap complete"
