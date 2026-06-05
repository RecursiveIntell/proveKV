# proveKV ↔ stack-ids + boundary-compiler — Integration Spec (Tier 1)

**Crate:** `prove-kv` (within `~/proveKV/` workspace)
**Integration Crates:** `stack-ids`, `boundary-compiler` (both in `~/Coding/Libraries/`)
**Author:** Rocky (provenance-first tier-1 audit, 2026-06-03)
**Status:** Draft, ready for execution
**Cost class:** Mechanical migration. One PR. Invalidates every published digest.
**Doctrine class:** Foundational. Do this *before* Tier 2 lands. Tier 2 receipts depend on it.

---

## 0. Why this spec exists

proveKV's `manifest.rs`, `pool.rs`, `receipt.rs`, and `shell.rs` each call
`blake3::hash(json.as_bytes()).to_hex()` on a `serde_json::to_string(self)` output.
That's 8 different `digest()` methods doing the same thing. The current pattern is
fine *until* a published receipt is contested — at which point the question "what
canonical form was this hashed over?" has a fragile answer:

- `serde_json::to_string` is **not canonical**. Map key ordering depends on
  `serde_json`'s `BTreeMap` vs `Map` decision per type.
- Rust struct field order is the *declaration* order — stable today, but not an
  enforced contract. A future refactor that reorders fields silently changes every
  digest.
- The same pattern exists in `semantic-memory` (own `canonical_json_string`),
  `forge-memory-bridge` (own canonicalizer), and `proveKV` (own ad-hoc). Three
  implementations of the same idea is one bug's worth of drift.

The fix: use `stack-ids::ContentDigest` (typed BLAKE3 wrapper) backed by
`boundary-compiler::Canonicalizer` (RFC 8785 JCS). This is the
cross-stack-consistent canonicalization path. It's already what the digest law in
`stack-ids::digest` docstring says is correct.

**The blocker is the cost, not the design.** Every published result in
`results/bench/.../state.json` says "this pool's digest is X." Switching canonicalizers
invalidates every X. The README's receipt tables cite those X values. The migration
is "re-run every published benchmark, regenerate every digest, update the README's
receipt tables." That's one PR, but it's a *public* PR — the README headline
numbers don't change (compression ratio, PPL delta), but the receipt fingerprints
they cite all do.

**Recommended posture:** do the migration **before** the next published batch of
results, not after. The next published batch will then be the first batch on the new
canonicalizer, and the README's "Receipts in `results/bench/...` reconcile to the
tables above" claim stays clean.

---

## 1. Scope

### In scope

- Replace 8 ad-hoc `digest()` methods in `prove-kv` with `stack-ids::ContentDigest`.
- Replace `serde_json::to_string(self)` serialization (in digest computation only) with
  `boundary_compiler::Canonicalizer::canonicalize(&serde_json::to_value(self)?)`.
- Add new ID newtypes to `stack-ids` and use them in proveKV's public API.
- Update `PoolBuildReceipt`, `ShellMaterializeReceipt`, `InjectionReceipt`,
  `PoolManifest`, `ShellManifest` field types from `String` to typed wrappers.
- Update `results/bench/.../state.json` files in the same PR (regenerated, not edited
  by hand).
- Update README's receipt tables to reflect new digests (compression ratios and PPL
  deltas do NOT change; only the digest hex values do).
- Add tests proving the canonicalization is RFC 8785 (cross-checked against a
  reference implementation if one is available; otherwise against a test vector
  suite).

### Out of scope

- `quant-codec-core`'s `CodecId` type (currently a `String` newtype). The
  codec id space is a different concern (codec identity, not digest identity)
  and would couple proveKV to a `quant-codec-core` evolution that isn't
  on the critical path.
- Receipt schema versions. The schema stays `prove_kv_receipt_v1` — only the
  digest values change.
- Wire format on disk (`FB1`, `TQW1`, `TQB1`, `TQB1-L` magic bytes). These are
  codec wire formats, not receipt digests. The on-disk format is read by the
  codec, not by the receipt layer.
- Tier 2/3 work (bitemporal timestamps, claim-ledger append, semantic-memory
  sidecar, quant-governor integration). Those depend on this landing first.

---

## 2. Dependency setup

Add to `proveKV/proveKV/Cargo.toml`:

```toml
[dependencies]
stack-ids = { path = "../../Coding/Libraries/stack-ids" }
boundary-compiler = { path = "../../Coding/Libraries/boundary-compiler" }
```

**Path question:** the proveKV workspace is at `~/proveKV/` and Libraries is at
`~/Coding/Libraries/`. The relative path `../../Coding/Libraries/stack-ids` is the
cleanest cross-workspace path. Alternative: vendor `stack-ids` and `boundary-compiler`
into the proveKV workspace as path deps. **Recommend path deps across workspaces** —
keeps the libraries' git history as the source of truth and avoids drift.

**Feature flag:** none needed at first. `stack-ids` and `boundary-compiler` are
foundational; no opt-in. If a future user wants proveKV without stack-ids (e.g.,
they want their own digest scheme), that becomes a `no_stack_ids` feature flag in a
follow-up. Not blocking.

---

## 3. New ID newtypes in `stack-ids`

`stack-ids` already exposes: `ArtifactId`, `EnvelopeId`, `ClaimId`, `EpisodeId`,
`AttemptId`, `TrialId`, `KernelRunId`, `ConstraintId`, `HyperedgeId`, `ResidualId`,
`SyndromeId`, `WitnessId`, `CertificateId`, `OracleSliceId`, `RefutationResultId`,
`OperatorId`, `OperatorVersionId`, `CalibrationReportId`, `ProjectionId`,
`RelationId`, `RelationVersionId`, `ImportBatchId`, `EntityId`, `RecordId`.

proveKV needs five new newtypes. The cleanest path is to add them to
`stack-ids/src/ids.rs` next to the existing ones:

```rust
// In stack-ids/src/ids.rs (or a new src/prove_kv_ids.rs module if prefer
// namespace separation)

/// Blake3 digest of a proveKV pool's content-addressed storage.
/// 64-char hex, computed via stack_ids::ContentDigest over the JCS-canonical
/// PoolManifest.
pub struct PoolId(pub ContentDigest);

/// Blake3 digest of a proveKV shell's content-addressed storage.
/// 64-char hex, computed via stack_ids::ContentDigest over the JCS-canonical
/// ShellManifest.
pub struct ShellId(pub ContentDigest);

/// Blake3 digest of a proveKV receipt's canonical serialization.
/// 64-char hex, computed via stack_ids::ContentDigest over the JCS-canonical
/// receipt (PoolBuildReceipt, ShellMaterializeReceipt, or InjectionReceipt).
pub struct ReceiptId(pub ContentDigest);

/// Blake3 digest of a fib-quant or turbo-quant codebook.
/// 64-char hex, computed via stack_ids::ContentDigest over the JCS-canonical
/// codebook representation (the serialized V1 codec profile + cluster centers).
pub struct CodebookId(pub ContentDigest);

/// Blake3 digest of a fib-quant or turbo-quant rotation matrix.
/// 64-char hex, computed via stack_ids::ContentDigest over the JCS-canonical
/// rotation (the serialized random orthogonal matrix used for the projection
/// step).
pub struct RotationId(pub ContentDigest);
```

**Why newtypes, not just `ContentDigest` everywhere:** prevents the field-swap bug
class. Right now `PoolBuildReceipt` has `pool_digest: String`, `codebook_digest: String`,
`rotation_digest: String` — all 64-char hex strings. Nothing in the type system
prevents `let x = receipt.codebook_digest.clone(); foo(pool_digest_slot, x);`. With
newtypes, that fails to compile.

**Schema versioning:** `stack-ids` newtypes are versionless. The schema version
goes on the receipt itself (`schema_version: String = "pool_build_receipt_v1"`).
A receipt is `ReceiptId` + the JCS-canonical of the receipt's schema version
*included in the digest*. So changing the schema version from `v1` to `v2` for
a fixed pool produces a different `ReceiptId` — which is the correct semantics.

**`Deref` to `&ContentDigest` vs owned `ContentDigest`:** recommend **owned**
`ContentDigest` inside the newtype. Avoids lifetime gymnastics on the public API.
`newtype(pub ContentDigest)` — caller does `pool_id.0.hex()` if they need the
hex string, or just `&pool_id.0` for `PartialEq`/`Hash`/`Display` impls that
delegate to `ContentDigest`.

**Trait impls needed:** `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Serialize`,
`Deserialize`, `Display`, `From<ContentDigest>`, `From<&str>` (parse),
`FromStr`, `AsRef<[u8]>`, `JsonSchema` (if `stack-ids` uses schemars).

---

## 4. Migration of proveKV digest computation

### 4.1 New helper in proveKV (private to crate)

In a new `prove-kv/src/digest.rs`:

```rust
use stack_ids::ContentDigest;
use boundary_compiler::Canonicalizer;

/// JCS-canonicalize a serde-serializable value and return its BLAKE3 digest
/// as a ContentDigest.
///
/// Panics on serialization failure (the only failure modes are programmer
/// errors — unserializable types — and we want those to fail loud at the
/// digest call site, not be swallowed).
pub fn canonicalize_and_digest<T: serde::Serialize>(value: &T) -> ContentDigest {
    let c = Canonicalizer::new();
    let val = serde_json::to_value(value)
        .expect("value must serialize to JSON for digest computation");
    c.canonicalize(&val)
        .map(|bytes| ContentDigest::compute(&bytes))
        .unwrap_or_else(|e| panic!("JCS canonicalization failed: {e}"))
}
```

### 4.2 Replace 8 digest() call sites

Files to modify:

| File | Line (approx) | Current code | New code |
|------|---------------|--------------|----------|
| `prove-kv/src/codec.rs` | 87 | `let payload_digest = blake3::hash(&encoded_payload).to_hex().to_string();` | unchanged (this is a hash of raw bytes, not a JSON-canonical digest — see §4.3) |
| `prove-kv/src/manifest.rs` | 99 | `Ok(blake3::hash(json.as_bytes()).to_hex().to_string())` | `Ok(canonicalize_and_digest(self).hex())` |
| `prove-kv/src/manifest.rs` | 189 | same as above | same as above |
| `prove-kv/src/pool.rs` | 104 | `Ok(blake3::hash(json.as_bytes()).to_hex().to_string())` | same as above |
| `prove-kv/src/receipt.rs` | 117 | same | same |
| `prove-kv/src/receipt.rs` | 196 | same | same |
| `prove-kv/src/receipt.rs` | 284 | same | same |
| `prove-kv/src/shell.rs` | 43 | same | same |

That's 7 JSON-canonical digest call sites (the codec.rs one is a raw-bytes hash of
the encoded payload, which is *correct as-is* — see §4.3).

**Retention of `digest() -> String` on receipt types:** the public API of
`PoolBuildReceipt::digest()` currently returns `String`. There are two paths:

- **Path A (do not break API):** keep the `String`-returning method, change its
  implementation to call `canonicalize_and_digest(self).hex()`. Callers don't
  notice.
- **Path B (breaking change):** change return type to `ReceiptId`. Callers must
  update.

**Recommended: Path A for this PR, Path B in the next major version.** The
breaking change is small but it's also unforced — no caller in the proveKV
workspace breaks. Save the API tightening for the Tier 2/3 lands when receipts
start flowing through `claim-ledger` and the type system upgrade pays for itself.

### 4.3 The raw-bytes hash at `codec.rs:87`

`codec.rs:87` hashes the *encoded payload bytes* (the codec wire format, e.g. FB1
or TQW1 or TQB1). This is **not** a JSON-canonical digest — it's a content hash
of opaque binary data. The right tool here is `blake3::hash(bytes)`, not JCS.
**Leave it alone.**

The new `CodebookId` (above) hashes the *codec profile metadata* (k, N, etc.),
which *is* JSON-canonicalizable. The encoded payload hash is a separate
identifier — it's the content-addressed storage handle for a single compressed
block. Different identity, different digest.

If we want to be extra-careful: rename `payload_digest` -> `block_content_hash`
to make the distinction clear. This is a field rename, which is *also* a
breaking change to anyone reading JSON receipts. **Recommend: leave the field
name, add a docstring note that it's a raw-bytes hash, not a JCS digest.**

### 4.4 The pool_digest computation (the one that ships in receipts)

The most-trafficked digest is `PoolBuildReceipt::pool_digest`. Currently it's
computed as `blake3::hash(canonical_json(pool_manifest))`. Post-migration:

```rust
// in PoolBuildReceipt::new() or wherever pool_digest is computed
let pool_digest = canonicalize_and_digest(&pool_manifest).hex();
```

The pool_manifest is itself a serde-serializable struct, so this works
mechanically. The new digest value will be different from the old (different
canonicalization algorithm = different bytes = different hash). That's the
expected behavior and the reason every published result needs regeneration.

---

## 5. Receipt field type upgrades

After §3 lands, optionally upgrade the receipt struct field types. This is the
"Path B / no-Path-B" decision:

```rust
// In prove-kv/src/receipt.rs

// BEFORE:
pub struct PoolBuildReceipt {
    pub schema_version: String,
    pub pool_digest: String,
    pub layer_digests: Vec<String>,
    pub codebook_digest: String,
    pub rotation_digest: String,
    // ...
}

// AFTER (optional, breaking):
pub struct PoolBuildReceipt {
    pub schema_version: String,
    pub pool_digest: ReceiptId,        // typed
    pub layer_digests: Vec<ReceiptId>, // typed
    pub codebook_digest: CodebookId,   // typed
    pub rotation_digest: RotationId,   // typed
    // ...
}
```

**Recommend Path A (just the digest computation) for the first PR.** Field-type
upgrade is a second PR. No field-type upgrade is *required* for the canonicalization
fix; it just makes the type system stronger.

**Exception: `layer_digests: Vec<String>`** — these are layer-level pool_digests.
After the migration they all become `ReceiptId` (or stay `String` and just
contain a `ReceiptId`'s hex). Recommend keeping them as `String` to minimize
friction; the *typed* upgrade is opt-in.

---

## 6. Regenerating published results

### 6.1 The state.json files in `results/bench/.../`

The Python PPL bench in `scripts/build_prove_kv_corpus.py` + the
`prove_kv_dynamic_cache_roundtrip` Rust example produce the `state.json` files.
These are the receipts that ship in the README's tables. After §4 lands:

1. The bench code (Python + Rust) must be re-run.
2. The new `state.json` files have new `pool_digest` values.
3. The README's receipt tables cite the new digests.
4. The compression ratios and PPL deltas do **not** change. Only the digest
   values do.

**Estimate of work:** one full re-run of the published bench suite. The
`reproduce.md` quickstart walks through this.

### 6.2 Cross-check: same input, same output

The first new state.json after migration must reconcile to a state.json produced
by the *previous* version on the same input. Specifically:

- PPL delta: must match exactly (or within float epsilon). The PPL is computed
  on the decompressed vectors, which is independent of the digest algorithm.
- Compression ratio: must match exactly. The on-disk size is independent of
  digest algorithm.
- Digest values: **must differ** (different canonicalization = different
  bytes hashed = different output). This is the *expected* difference and is
  the whole point of the migration.

If a regenerated state.json has the *same* digest as the previous one, something
is wrong (the canonicalization is identical to `serde_json::to_string`, which
means JCS isn't actually running). Test for this explicitly.

---

## 7. Test plan

### 7.1 Unit tests (in `prove-kv/src/digest.rs` or a new `tests/digest.rs`)

```rust
#[test]
fn canonicalize_is_jcs_compliant() {
    // JCS requires: object keys in lexicographic order
    let c = Canonicalizer::new();
    let v = serde_json::json!({"b": 2, "a": 1});
    let bytes = c.canonicalize(&v).unwrap();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), r#"{"a":1,"b":2}"#);
}

#[test]
fn digest_is_stable_across_key_order() {
    let a = serde_json::json!({"a": 1, "b": 2, "c": 3});
    let b = serde_json::json!({"c": 3, "b": 2, "a": 1});
    assert_eq!(canonicalize_and_digest(&a), canonicalize_and_digest(&b));
}

#[test]
fn digest_uses_blake3() {
    // Reference BLAKE3 of the empty string
    let empty_blake3 = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
    let d = ContentDigest::compute(b"");
    assert_eq!(d.hex(), empty_blake3);
}

#[test]
fn digest_changes_when_field_reorders() {
    // Two receipts with the same fields but different declaration order
    // should produce different digests ONLY if the field is in the digest
    // domain. For a typed receipt, all fields are in the digest domain.
    let r1 = PoolBuildReceipt { /* fields in declaration order */ };
    let r2 = PoolBuildReceipt { /* same fields, but constructed via
                                   serde_json::from_value with reordered
                                   keys — must be different if JCS is
                                   the canonicalizer */ };
    // ... but this test is structural and may need adjustment
}

#[test]
fn pool_digest_matches_pool_manifest_digest() {
    // The pool's "digest" should equal the digest of its PoolManifest.
    // Cross-check this invariant: if it breaks, the receipt is lying.
    let manifest = PoolManifest { /* ... */ };
    let receipt = build_pool_and_get_receipt(/* ... */);
    assert_eq!(receipt.pool_digest, manifest.digest());
}
```

### 7.2 Conformance tests (RFC 8785 vectors)

If `boundary-compiler` already has an RFC 8785 conformance test suite, run it.
If not, port the [RFC 8785 Appendix A test vectors](https://www.rfc-editor.org/rfc/rfc8785)
into a `tests/rfc8785_vectors.rs`. This is the test that says "we are RFC 8785
compliant, not just 'I think this is canonical.'"

### 7.3 Cross-stack consistency test

The whole point of `stack-ids` is cross-crate consistency. Add a test in
`stack-ids` that says: "if `prove-kv` and `semantic-memory` both digest the
same conceptual object (e.g., a `Claim` with the same content), they produce
the same digest." This is a *contract* test, not a runtime test — it lives in
`stack-ids` and proveKV/semantic-memory each have a test that their digest
computation is wired to `stack_ids::ContentDigest`.

---

## 8. Migration order (the actual sequence of commits)

1. **Commit 1 — add new ID newtypes to `stack-ids`.** No callers yet. CI green.
2. **Commit 2 — add `canonicalize_and_digest` helper to `prove-kv`.** No callers
   yet. CI green.
3. **Commit 3 — replace `digest()` implementations in prove-kv.** The 7
   call sites in §4.2. CI green. Old digests change to new digests internally
   but no published artifacts change yet.
4. **Commit 4 — re-run the published bench suite.** New state.json files in
   `results/bench/.../`. CI green (bench numbers don't change).
5. **Commit 5 — update README receipt tables.** New digest values, same
   compression/PPL numbers. CI green.

Five commits, each individually revertable. If commit 3 breaks a test, you
revert to commit 2 and debug. If commit 4 produces different PPL (which would
be alarming), you revert to commit 3 and dig in. If commit 5 is just a
markdown edit, the worst case is a typo.

---

## 9. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| `boundary-compiler` JCS has a bug that produce non-canonical output | High — would silently make every digest wrong | Run RFC 8785 Appendix A vectors. Cross-check with `serde_json::to_string` for non-Map-key tests. |
| Some receipt type contains a non-JCS-canonicalizable field (e.g., a `f32` with NaN, or a `HashMap<String, T>` with non-string keys) | Medium | Audit all receipt fields before commit 3. The proveKV types are all `String`/`u32`/`u64`/`i64`/`f64`/`bool`/`Vec<X>` — all JCS-safe. If a `HashMap` is used, switch to `BTreeMap` or `IndexMap`. |
| Published `results/bench/.../state.json` files have a `pool_digest` that the README also cites | High — README is the public face | The README's tables cite both compression ratios AND pool_digest values. After regeneration, the compression ratios stay constant, the digests all change. The README update is mechanical (find-replace on the digest column). |
| A test depends on the *exact* `serde_json::to_string` output (e.g., a snapshot test) | Low | Inventory snapshot tests before commit 3. The proveKV workspace has a `tests/batched_wire_receipt.rs` — audit it. |
| The cross-workspace path dep (`../../Coding/Libraries/...`) breaks CI | Low | Set up a `cargo metadata` step in CI to verify the path. If CI runs in a container, mount both repos. Alternative: vendor `stack-ids` and `boundary-compiler` as path deps inside `proveKV/` workspace. |
| A future `serde_json` upgrade changes `to_string` output for `f64` (e.g., NaN handling) | Low — JCS doesn't use `serde_json::to_string` | We're using JCS, not `serde_json::to_string`. JCS canonicalization is well-defined for IEEE 754 floats per the RFC. |
| A receipt field is renamed and the digest changes silently | Medium | Schema version field on the receipt (already exists: `schema_version: String`). When the schema changes, bump the version. The digest is over the *new* schema, so consumers detect the change via the version field. |

---

## 10. Acceptance gates

This PR is done when **all** of the following hold:

- [ ] `stack-ids` exposes `PoolId`, `ShellId`, `ReceiptId`, `CodebookId`, `RotationId`.
- [ ] `boundary-compiler` RFC 8785 Appendix A test vectors pass.
- [ ] `prove-kv`'s 7 JSON-canonical digest call sites use `canonicalize_and_digest`.
- [ ] `cargo test -p prove-kv` is green.
- [ ] `cargo test --workspace` in `~/proveKV/` is green.
- [ ] Every `state.json` in `results/bench/.../` is regenerated and the PPL
      deltas reconcile to within float epsilon to the previously-published values.
- [ ] Every `pool_digest` in `results/bench/.../state.json` is different from
      the previously-published value (this is the explicit "we changed canonicalizers"
      check).
- [ ] README's receipt tables cite the new digests.
- [ ] README's headline compression ratios and PPL deltas are unchanged.
- [ ] A cross-stack consistency test exists in `stack-ids` showing `prove-kv` and
      `semantic-memory` produce the same digest for the same input.

---

## 11. What this spec does NOT cover

- The 5 new ID newtypes in `stack-ids` need their own PR description in the
  `stack-ids` repo (or wherever `stack-ids` lives). This spec assumes that PR
  is in flight alongside the proveKV PR.
- The `boundary-compiler` crate is a *dependency* of `stack-ids`, so the
  `stack-ids` PR will pull it in transitively. If `boundary-compiler` needs
  its own improvements (e.g., better error messages, support for non-string
  map keys), that's a separate PR to `boundary-compiler`.
- Migration of `semantic-memory`'s `canonical_json_string` to `boundary-compiler`
  is a *separate* Tier 1-ish work item. It's parallel to this spec, not in
  series. Doing both in one PR would be too big.
- Anything in Tier 2 or Tier 3.

---

## 12. Open questions for Josh

1. **Path dep vs. vendor:** do you want `stack-ids` and `boundary-compiler` as
   cross-workspace path deps (my recommendation) or vendored into `proveKV/`?
   The trade-off is git history cleanliness (path) vs. CI independence (vendor).
2. **Path B (typed field upgrade):** in this PR or a follow-up? I recommend
   follow-up, but if Tier 2 lands in the same sprint, doing it now saves a
   round-trip.
3. **Cross-stack consistency test location:** `stack-ids` (doctrine) or
   `prove-kv` (impl)? I lean `stack-ids` since it's the contract.
4. **README receipt tables:** updated by hand or generated from
   `results/bench/.../state.json`? Generated is more honest (the README is
   always in sync) but requires a build step. If the README is currently
   hand-edited, that's a separate work item.
5. **Pre-1.0 vs. post-1.0:** is this PR landing before or after proveKV's 1.0
   release? If after 1.0, the README's "Reproduce" section can promise
   "digests will be stable from this version forward" — which is a stronger
   claim than the current "digests are what they are" posture.

---

**End of Tier 1 spec.**
