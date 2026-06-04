# proveKV Publish Recovery Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
> Resume context: previous session yanked broken `poly-kv 0.1.0-alpha.1` and `fib-quant 0.1.0-alpha.2` from crates.io on 2026-06-03. `~/proveKV/` has 2 unpushed local commits and 8 dirty files; build is green, all tests pass.

**Goal:** Restore a clean, correct, publishable crate graph for proveKV: provekv (lib provekv, display proveKV) published fresh from `~/proveKV/proveKV/`, with fib-quant and turbo-quant as registry deps at versions that actually have the APIs the local code uses.

**Architecture:** Three-step publish order (turbo-quant UPDATE → fib-quant UPDATE → prove-kv rename + first publish). Each publish only after its full workspace builds and tests pass. Repo URLs stay correct: turbo-quant → `RecursiveIntell/turbo-quant`, provekv → `RecursiveIntell/proveKV`. **No yanks of any turbo-quant version** — those are live with real downloads and article references.

**Tech Stack:** Rust 1.95, cargo 1.95, crates.io, fib-quant/turbo-quant/prove-kv workspace.

**Validated pre-publish gate (as of 2026-06-03):** `cargo check --workspace --all-targets --all-features` clean, `cargo test --workspace --all-targets --all-features` green (0 failed). `cargo clippy --workspace --all-targets --all-features -- -D warnings` has 7 pre-existing lints (gpu-backend 5, turbo-quant 2) — the project's actual release gate is test pass, not clippy clean. Note this in CHANGELOG per release.

---

## State Snapshot (as of 2026-06-03, verified)

| Item | Value |
|---|---|
| `~/proveKV/` branch | `main`, 2 commits ahead of `origin/main` (`9487a55 Visuals: address reviewer feedback`, `a99229c README + visuals: lead with PPL-validated 37.31x / 65.88x headline`) |
| Working tree | 8 files modified, untracked `.hermes/`. ~334 lines added across `proveKV/` (5 files) and `turbo-quant/` (3 files) |
| Workspace members | `fib-quant`, `proveKV`, `quant-codec-core`, `gpu-backend`, `turbo-quant` |
| Canonical crate | `prove-kv` (lib `prove_kv`, display `proveKV`) at `~/proveKV/proveKV/` — TO BE RENAMED to `provekv` |
| Repo URL in local Cargo.toml | `https://github.com/RecursiveIntell/proveKV` (correct) |
| crates.io state | `poly-kv 0.1.0-alpha.1` YANKED, `fib-quant 0.1.0-alpha.2` YANKED, `turbo-quant 0.1.0/0.2.0/0.2.1` LIVE (downloads 2893/18/18) |
| Compile state | GREEN. `cargo check --workspace --all-targets --all-features` clean. |
| Test state | GREEN. `cargo test --workspace --all-targets --all-features` 0 failed. |
| Clippy state | 7 pre-existing lints, project's gate is test pass. |

### Local crate versions vs. crates.io

| Crate | Local | crates.io | Notes |
|---|---|---|---|
| `turbo-quant` | 0.2.0 | 0.2.1 (live) | Local is BEHIND crates.io. Decide next version (0.2.2 additive or 0.3.0 breaking). |
| `fib-quant` | 0.1.0-alpha.1 | 0.1.0-alpha.2 (yanked) | Local is BEHIND yanked version. Need to bump to 0.1.0-alpha.2 (now free after yank) or 0.1.0-alpha.3 (next available) with the new batch API. |
| `prove-kv` | 0.1.0-alpha.1 | never published under this name | First publish after rename to `provekv` and version bump to 0.1.0-alpha.2. |
| `quant-codec-core` | 0.1.0-alpha.1 | 0.1.0-alpha.1 (live) | No changes needed unless prove-kv refactor touches it. |
| `gpu-backend` | 0.1.0-alpha.1 | 0.1.0-alpha.1 (live) | Same. |

### turbo-quant crates.io detail (verified 2026-06-03 via API)

- `0.1.0` — 2026-03-26, 2,893 downloads, repo `RecursiveIntell/turbo-quant`, NOT yanked. Mentioned in articles.
- `0.2.0` — 2026-05-19, 18 downloads, NOT yanked.
- `0.2.1` — 2026-06-02 23:42 UTC, 18 downloads, NOT yanked. Published by RecursiveIntell/Josh Stevenson.

**DO NOT yank any turbo-quant version.** Live downloads + article references make this destructive and visible. Next turbo-quant release is an UPDATE shipping the uncommitted work in `~/proveKV/turbo-quant/src/{polar,rotation,turbo}.rs` (TQB1 batched wire format, rotation matrix, batch turbo quantizer).

### fib-quant crates.io detail (verified 2026-06-03 via API)

- `0.1.0-alpha.1` — live. Released before the encode_batch/decode_batch API.
- `0.1.0-alpha.2` — YANKED 2026-06-03. Was the version with TQW1 wire format but missing the batch encode/decode methods.
- Next available: `0.1.0-alpha.2` (yanked version string is now free) or `0.1.0-alpha.3` (if semantic intent is "next iteration").

### 8 dirty files in the working tree

- `proveKV/Cargo.toml` — 6 lines
- `proveKV/examples/prove_kv_multi_agent_shell.rs` — 35 lines
- `proveKV/src/codec.rs` — 31 lines
- `proveKV/src/policy.rs` — 120 lines (TQB1 batched codec id matching for bits 2..=16)
- `proveKV/src/shell.rs` — 22 lines
- `turbo-quant/src/polar.rs` — 79 lines
- `turbo-quant/src/rotation.rs` — 81 lines (new `StoredRotation` / `FastHadamardRotation` types)
- `turbo-quant/src/turbo.rs` — 13 lines

---

## Task 1: Lock the publish order and version decisions

**Objective:** Resolve open naming/version questions before writing any code.

**Step 1: Confirm crate name — DECIDED 2026-06-03**

Publish canonical code as `provekv` (lowercase, no hyphen).

**Step 2: Confirm crates.io name — DECIDED 2026-06-03**

The yank left `poly-kv` available. **DECIDED:** publish as `provekv` (lowercase, no hyphen). The user confirmed this with a single-word reply: "provekv."

Implications for Task 5 (rename):
- Cargo.toml `name = "prove-kv"` → `name = "provekv"`
- Library path `name = "prove_kv"` → `name = "provekv"` (Rust crate identifiers can't have hyphens, so the lib name mirrors the package name as a single word)
- `use prove_kv::...` in src/lib.rs, examples/*.rs → `use provekv::...`
- `RECEIPT_SCHEMA = "prove_kv_receipt_v1"` in src/receipt.rs: bump to `"provekv_receipt_v1"` ONLY if the receipt format changes. If only the identifier changes, this is a wire-format break — note in CHANGELOG and require decode-side handling for old receipts.
- README.md `use prove_kv::...` → `use provekv::...`
- GitHub repo name stays `RecursiveIntell/proveKV` (display name unchanged). The crates.io name and GitHub repo name intentionally diverge to match the user's stated preference.

**Step 3: Decide fib-quant version bump**

Local uncommitted work in `~/proveKV/fib-quant/src/codec.rs` adds `encode_batch`/`decode_batch`. Decide:
- `0.1.0-alpha.2` (now free after yank) — straightforward, version is the obvious "next"
- `0.1.0-alpha.3` — if semantic intent is "we've now iterated twice on this"
- `0.2.0` (drop alpha) — only if API is stable

The yanked version is gone forever, so `0.1.0-alpha.2` is the cleanest reuse.

**Step 4: Decide turbo-quant version — UPDATE, not first publish**

turbo-quant 0.1.0/0.2.0/0.2.1 are LIVE on crates.io (verified 2026-06-03, downloads 2893/18/18). Local `~/proveKV/turbo-quant/Cargo.toml` is at `0.2.0`. Uncommitted work in `~/proveKV/turbo-quant/src/{polar,rotation,turbo}.rs` is the TQB1 batched wire format + rotation + batch turbo quantizer.

Decide the next version:
- `0.2.2` if TQB1 is purely additive (new module/function on existing struct, backward compatible)
- `0.3.0` if TQB1 changes existing public API in a breaking way

Inspect the diff to make the call. **Never yank a turbo-quant version** — live downloads + article references make it destructive and visible.

**Step 5: Write the decision ledger**

Append a `PUBLISH_DECISIONS.md` to `~/proveKV/` recording the answers to Steps 1-4. This is the source of truth for the next session.

**Step 6: Commit the ledger only**

```bash
cd ~/proveKV
git add PUBLISH_DECISIONS.md
git commit -m "docs: record crates.io publish decisions for 0.1.0-alpha.2 series"
```

---

## Task 2: Push the 2 unpushed commits + commit the dirty tree

**Objective:** Get the working tree to a clean, pushed state with green tests.

**Files:**
- 2 unpushed commits: `9487a55`, `a99229c`
- 8 dirty files (listed above)

**Step 1: Push the unpushed commits first**

The 2 unpushed commits are `Visuals: address reviewer feedback` and `README + visuals: lead with PPL-validated 37.31x / 65.88x headline`. They are your work since the last push. Push them before doing anything else so we have a clean baseline.

```bash
cd ~/proveKV
git push origin main
```

If push fails (auth, network), STOP and report. Do not proceed to commits.

**Step 2: Re-verify green after push**

```bash
cd ~/proveKV
cargo test --workspace --all-targets --all-features
```

Gate: still 0 failed. If anything breaks from the push (unlikely, but defensive), stop.

**Step 3: Stage the 8 dirty files**

```bash
cd ~/proveKV
git add proveKV/Cargo.toml proveKV/examples/prove_kv_multi_agent_shell.rs \
        proveKV/src/codec.rs proveKV/src/policy.rs proveKV/src/shell.rs \
        turbo-quant/src/polar.rs turbo-quant/src/rotation.rs turbo-quant/src/turbo.rs
git status --short
```

Confirm 8 files staged, 0 unstaged (other than `.hermes/` which is intentionally untracked).

**Step 4: Commit in 2 logical groups**

Split by crate so each commit is reviewable:

```bash
cd ~/proveKV
git commit -m "feat(turbo-quant): TQB1 batched wire format, rotation matrices, batch turbo quantizer"
git commit -m "feat(prove-kv): TQB1 policy matching for 2..=16 bit rates, multi-agent shell"
```

**Step 5: Run the full test suite again after each commit**

```bash
cd ~/proveKV
cargo test --workspace --all-targets --all-features
cargo test --workspace --doc
```

Gate: every commit must leave the workspace green.

---

## Task 3: Publish fib-quant with the new batch API

**Objective:** Get the version of fib-quant with `encode_batch`/`decode_batch` onto crates.io so provekv can depend on it via registry (not path).

**Files:**
- `~/proveKV/fib-quant/Cargo.toml` (version bump)
- `~/proveKV/fib-quant/src/lib.rs` (re-export the new API)

**Step 1: Bump version in Cargo.toml**

Per Task 1 decision, set `version` in `~/proveKV/fib-quant/Cargo.toml`. Update the `CHANGELOG.md` if present. **Recommended: 0.1.0-alpha.2** (the yanked version string is now free).

**Step 2: Verify no path deps remain**

fib-quant should have no internal workspace path deps. Confirm.

**Step 3: Dry-run**

```bash
cd ~/proveKV/fib-quant
cargo publish --dry-run 2>&1 | tail -30
```

Expected: clean tarball, no errors, no path-dep warnings.

**Step 4: Publish**

```bash
cd ~/proveKV/fib-quant
cargo publish 2>&1 | tail -10
```

Expected: `Uploading fib-quant v0.1.0-alpha.2` then `Uploaded`.

**Step 5: Verify on crates.io**

```bash
curl -sS -A "github.com/RecursiveIntell/proveKV (audit)" "https://crates.io/api/v1/crates/fib-quant/0.1.0-alpha.2" -o /tmp/verify-fib.json
python3 -c "
import json
d = json.load(open('/tmp/verify-fib.json'))
v = d.get('version', {})
print('fib-quant 0.1.0-alpha.2:')
print('  yanked:', v.get('yank'))
print('  downloads:', v.get('downloads'))
"
```

Wait 30-60s for the registry index to propagate before continuing.

---

## Task 4: Publish turbo-quant

**Objective:** Update turbo-quant on crates.io to ship the TQB1 work.

**Files:**
- `~/proveKV/turbo-quant/Cargo.toml` (version bump)
- `~/proveKV/turbo-quant/README.md` (verify still accurate)

**Step 1: Bump version in Cargo.toml**

Per Task 1 decision. The repo URL stays `RecursiveIntell/turbo-quant` (separate from proveKV).

**Step 2: Verify no path deps remain**

turbo-quant should have no internal workspace path deps that would block publish. Confirm.

**Step 3: Dry-run**

```bash
cd ~/proveKV/turbo-quant
cargo publish --dry-run 2>&1 | tail -30
```

**Step 4: Publish**

```bash
cd ~/proveKV/turbo-quant
cargo publish 2>&1 | tail -10
```

**Step 5: Verify**

```bash
curl -sS -A "github.com/RecursiveIntell/proveKV (audit)" "https://crates.io/api/v1/crates/turbo-quant/<NEW_VERSION>" -o /tmp/verify-tq.json
python3 -c "
import json
d = json.load(open('/tmp/verify-tq.json'))
v = d.get('version', {})
print('turbo-quant <NEW_VERSION>:')
print('  yanked:', v.get('yank'))
print('  downloads:', v.get('downloads'))
"
```

---

## Task 5: Rename prove-kv → provekv and update to registry deps

**Objective:** Apply the rename decision from Task 1 across the workspace and switch path deps to registry deps.

**Files:**
- `~/proveKV/proveKV/Cargo.toml` (name, lib name, dep versions)
- `~/proveKV/proveKV/src/lib.rs` (doc examples)
- `~/proveKV/proveKV/src/receipt.rs` (RECEIPT_SCHEMA wire-format string — bump to v2 only if format actually changes)
- `~/proveKV/proveKV/examples/*.rs` (3 example files use `use prove_kv::...`)
- `~/proveKV/proveKV/README.md` (use statements in docs)

**Step 1: Search for all references**

```bash
cd ~/proveKV/proveKV
grep -rn "prove_kv\|prove-kv" --include="*.rs" --include="*.toml" --include="*.md"
```

Confirm the surface matches what was discovered earlier: 2 lines in Cargo.toml, 1 in src/lib.rs, 1 in src/receipt.rs, 3+ example files, README.

**Step 2: Apply renames**

Use `patch` (fuzzy match) for each occurrence. Be careful with the `RECEIPT_SCHEMA` string — that's a wire identifier. Bump to `poly_kv_receipt_v2` (or whatever the new name is) ONLY if the receipt format also changes. Otherwise leave it as a wire-format break and call it out in CHANGELOG.

**Step 3: Update path deps to registry deps**

In `~/proveKV/proveKV/Cargo.toml`:
```toml
fib-quant = { version = "0.1.0-alpha.2", optional = true }
turbo-quant = { version = "<new-turbo-quant-version>", optional = true }
```

Drop the `path = "..."` entries. Add `registry = "crates-io"` only if not the default registry.

**Step 4: Bump version to 0.1.0-alpha.2**

The current `0.1.0-alpha.1` is not on crates.io (it was always the wrong `poly-kv` that was published). Bumping is defensive — prevents any future confusion.

**Step 5: Compile and test**

```bash
cd ~/proveKV/proveKV
cargo check --all-features
cargo test --all-features
```

Gate: green, no warnings introduced.

---

## Task 6: Publish the canonical crate as `provekv`

**Objective:** Get the renamed, registry-dep'd crate onto crates.io under the agreed name.

**Step 1: Dry-run with all features**

```bash
cd ~/proveKV/proveKV
cargo publish --dry-run --all-features 2>&1 | tail -40
```

If clean, the previous `cargo publish --dry-run` (which failed because of unpublished fib-quant API) should now succeed.

**Step 2: Sanity check the metadata**

Before publishing, read the would-be package metadata:
```bash
cd ~/proveKV/proveKV
cargo package --list
tar -tzf target/package/provekv-0.1.0-alpha.2.crate 2>&1 | head -30
```

Confirm:
- `Cargo.toml` has `repository = "https://github.com/RecursiveIntell/proveKV"` (NOT Libraries)
- No path deps left
- No LICENSE file missing
- No giant fixture files bloating the tarball

**Step 3: Tag the release in git**

```bash
cd ~/proveKV
git tag -a v0.1.0-alpha.2 -m "publish 0.1.0-alpha.2 to crates.io"
git push origin v0.1.0-alpha.2
```

**Step 4: Publish**

```bash
cd ~/proveKV/proveKV
cargo publish 2>&1 | tail -10
```

Expected: `Uploading provekv v0.1.0-alpha.2` → `Uploaded`.

**Step 5: Verify on crates.io**

```bash
curl -sS -A "github.com/RecursiveIntell/proveKV (audit)" "https://crates.io/api/v1/crates/provekv/0.1.0-alpha.2" -o /tmp/verify-provekv.json
python3 -c "
import json
d = json.load(open('/tmp/verify-provekv.json'))
v = d.get('version', {})
print('provekv 0.1.0-alpha.2:')
print('  yanked:', v.get('yank'))
print('  downloads:', v.get('downloads'))
print('  repository:', v.get('repository'))
"
```

Then visually confirm in browser that the repo link goes to proveKV, not Libraries.

**Step 6: Smoke test as a consumer**

In a scratch dir, build a tiny binary that depends on the freshly published crate:

```bash
mkdir -p /tmp/provekv-smoke && cd /tmp/provekv-smoke && git init
cat > Cargo.toml <<'EOF'
[package]
name = "provekv-smoke"
version = "0.0.1"
edition = "2021"

[dependencies]
provekv = "0.1.0-alpha.2"
EOF
mkdir src
cat > src/main.rs <<'EOF'
fn main() {
    println!("smoke test for provekv");
}
EOF
cargo build --release
```

Expected: clean build, registry resolution works, no path errors. This is the only test that proves the published crate is actually consumable.

---

## Task 7: Update documentation

**Objective:** Reflect the new crates.io state in the repo docs.

**Files:**
- `~/proveKV/README.md`
- `~/proveKV/CHANGELOG.md` (create if absent)
- `~/proveKV/CITATION.cff` (if applicable)

**Step 1: Update README install instructions**

Replace any reference to `poly-kv = "0.1.0-alpha.1"` with `provekv = "0.1.0-alpha.2"`. Add a brief note: "This crate was previously published as `poly-kv 0.1.0-alpha.1`; that version has been yanked. The canonical crate is now `provekv 0.1.0-alpha.2`."

**Step 2: Write CHANGELOG entries**

```markdown
## fib-quant 0.1.0-alpha.2 — 2026-XX-XX
### Added
- `encode_batch` / `decode_batch` for SIMD+Rayon acceleration across N inputs

## turbo-quant 0.2.2 — 2026-XX-XX
### Added
- TQB1 batched wire format (polar + turbo)
- `StoredRotation` and `FastHadamardRotation` types
- Batch turbo quantizer

## provekv 0.1.0-alpha.2 — 2026-XX-XX
### Fixed
- Published under correct crate name; previous `poly-kv 0.1.0-alpha.1` had wrong repository URL.
- Now depends on registry versions of fib-quant (with `encode_batch`/`decode_batch`) and turbo-quant (with TQB1) instead of workspace path deps.

### Added
- 17.17x at N=8 multi-agent hot-tier sweep
- TQB1 compact wire format policy matching for 2..=16 bit rates
```

**Step 3: Update CITATION.cff if present**

The version field and any URL pointing to crates.io should reflect the new state.

**Step 4: Commit and push**

```bash
cd ~/proveKV
git add README.md CHANGELOG.md CITATION.cff
git commit -m "docs: update for 0.1.0-alpha.2 series publish"
git push origin main
```

---

## Phase Gates

After every task, run:

```bash
cd ~/proveKV
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-targets --all-features
cargo publish -p <CRATE> --dry-run
```

A task is not done until all of those are clean. Block on the user if any fails.

**Clippy is intentionally NOT a hard gate.** The repo's published 0.2.0/0.2.1 turbo-quant has 7 pre-existing clippy warnings (gpu-backend 5, turbo-quant 2) that the previous release didn't fix. The project's actual gate is `cargo test`. Note the pre-existing clippy state in the CHANGELOG and move on.

---

## Rollback / Quarantine

- If Task 6 publish succeeds but the smoke test fails: the crate is technically published but not consumable. YANK it. Don't try to fix in place — yank, debug, re-publish at a new version.
- If Task 3 or Task 4 publish succeeds and the next consumer fails: don't yank the upstream crate; instead fix the consumer's Cargo.toml and re-test.
- The two yanks from 2026-06-03 (`poly-kv 0.1.0-alpha.1`, `fib-quant 0.1.0-alpha.2`) are unrecoverable — those version strings are reserved forever. Plan around the constraint; don't try to "fix" the yank state.
