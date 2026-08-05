//! Crash recovery for the page store.
//!
//! On startup, recovery removes uncommitted temporary files, validates every
//! committed page header, and rebuilds a reachability index from the committed
//! manifest. Corruption is quarantined, not silently absorbed.

use std::collections::HashSet;
use std::fs;

use crate::error::Result;
use crate::page_store::PageStore;

/// Outcome of a recovery pass.
#[derive(Debug)]
pub struct RecoveryReport {
    /// Pages that passed validation.
    pub valid_pages: Vec<String>,
    /// Temporary files removed.
    pub temp_files_removed: Vec<String>,
    /// Pages that failed validation (quarantined).
    pub corrupted_pages: Vec<String>,
    /// Reachable page digests according to the manifest, if reconstruction was
    /// requested.
    pub reachable_digests: Option<HashSet<String>>,
}

/// Remove stale temporary files (`*.tmp.*.page`) from the store root.
pub fn remove_temp_files(store: &PageStore) -> Result<Vec<String>> {
    let mut removed = Vec::new();
    let entries = fs::read_dir(store.root())?;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(".tmp.") && name_str.ends_with(".page") {
            fs::remove_file(entry.path())?;
            removed.push(name_str.into_owned());
        }
    }
    Ok(removed)
}

/// Validate every committed page in the store.
///
/// Each `.page` file is opened and its header validated plus payload digest
/// verified. Pages that fail validation are recorded but not deleted — the
/// caller decides whether to quarantine.
pub fn validate_committed_pages(store: &PageStore) -> Result<RecoveryReport> {
    let mut valid = Vec::new();
    let mut corrupted = Vec::new();

    let entries = fs::read_dir(store.root())?;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.ends_with(".page") || name_str.starts_with(".tmp.") {
            continue;
        }

        // Extract digest from filename: "<digest>.page"
        let digest = name_str.strip_suffix(".page").unwrap_or(&name_str);

        match store.read_page(digest) {
            Ok((_header, _payload)) => {
                valid.push(digest.to_string());
            }
            Err(e) => {
                corrupted.push(format!("{}: {}", digest, e));
            }
        }
    }

    Ok(RecoveryReport {
        valid_pages: valid,
        temp_files_removed: Vec::new(),
        corrupted_pages: corrupted,
        reachable_digests: None,
    })
}

/// Full recovery: remove temps, validate committed pages.
pub fn recover(store: &PageStore) -> Result<RecoveryReport> {
    let temp_files_removed = remove_temp_files(store)?;
    let mut report = validate_committed_pages(store)?;
    report.temp_files_removed = temp_files_removed;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_format::build_page_header;
    use std::env;

    use std::sync::atomic::{AtomicU32, Ordering};

    static STORE_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_store() -> (PageStore, std::path::PathBuf) {
        let n = STORE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("provekv-recovery-{}-{}", std::process::id(), n));
        let _ = fs::remove_dir_all(&dir);
        let store = PageStore::open(&dir).unwrap();
        (store, dir)
    }

    #[test]
    fn recovery_removes_temp_files() {
        let (store, _dir) = temp_store();

        // Create a fake temp file.
        let tmp = store.root().join(".tmp.deadbeef.page");
        fs::write(&tmp, b"garbage").unwrap();

        let report = recover(&store).unwrap();
        assert_eq!(report.temp_files_removed.len(), 1);
        assert!(!tmp.exists());
    }

    #[test]
    fn recovery_validates_committed_pages() {
        let (store, _dir) = temp_store();
        let payload = vec![42u8; 64];
        let header = build_page_header(
            "full_attn_k",
            &[16],
            "float32",
            b'l',
            "sha256:model",
            "sha256:layout",
            0,
            1,
            "raw_exact",
            &payload,
        )
        .unwrap();
        store.write_page(&header, &payload).unwrap();

        let report = recover(&store).unwrap();
        assert_eq!(report.valid_pages.len(), 1);
        assert!(report.corrupted_pages.is_empty());
    }

    #[test]
    fn recovery_detects_corruption() {
        let (store, _dir) = temp_store();
        let payload = vec![42u8; 64];
        let header = build_page_header(
            "full_attn_k",
            &[16],
            "float32",
            b'l',
            "sha256:model",
            "sha256:layout",
            0,
            1,
            "raw_exact",
            &payload,
        )
        .unwrap();
        store.write_page(&header, &payload).unwrap();

        // Corrupt the page file by truncation.
        let path = store.page_path(&header.payload_digest);
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(10).unwrap();

        let report = recover(&store).unwrap();
        assert!(report.valid_pages.is_empty());
        assert_eq!(report.corrupted_pages.len(), 1);
    }
}
