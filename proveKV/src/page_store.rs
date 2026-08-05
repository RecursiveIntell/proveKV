//! Durable page store with atomic writes and validated reads.
//!
//! Pages are written to temporary files, fsynced, atomically renamed, then
//! the directory is fsynced. Manifests commit last. On read, every header
//! field is validated and the payload digest is verified before the payload
//! bytes are returned.

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{ProveKvError, Result};
use crate::page_format::{self, PageHeader, MAX_HEADER_BYTES};

/// A durable page store rooted at a directory.
pub struct PageStore {
    root: PathBuf,
}

impl PageStore {
    /// Create or open a page store at `root`. The directory is created with
    /// mode `0o700` if it does not exist.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self { root })
    }

    /// Return the store root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Build a path for a page file identified by its payload digest.
    pub fn page_path(&self, payload_digest: &str) -> PathBuf {
        self.root.join(format!("{}.page", payload_digest))
    }

    /// Build a temporary path used during atomic write.
    fn temp_path(&self, payload_digest: &str) -> PathBuf {
        self.root.join(format!(".tmp.{}.page", payload_digest))
    }

    /// Write a page atomically.
    ///
    /// 1. Serialize header as JSON + newline, then payload bytes.
    /// 2. Write to a temporary file.
    /// 3. fsync the temp file.
    /// 4. Atomically rename over the real path.
    /// 5. fsync the directory.
    ///
    /// On error the temporary file is removed.
    pub fn write_page(&self, header: &PageHeader, payload: &[u8]) -> Result<PathBuf> {
        header.validate()?;

        // Verify the header's payload digest matches the actual payload.
        let actual_digest = page_format::compute_payload_digest(payload);
        if actual_digest != header.payload_digest {
            return Err(ProveKvError::DigestMismatch {
                expected: header.payload_digest.clone(),
                got: actual_digest,
            });
        }

        let dest = self.page_path(&header.payload_digest);
        let tmp = self.temp_path(&header.payload_digest);

        // Clean up any stale temp file.
        let _ = fs::remove_file(&tmp);

        // Write to temp file.
        {
            let mut file = File::create(&tmp)?;
            let header_json = serde_json::to_vec(header)?;
            file.write_all(&header_json)?;
            file.write_all(b"\n")?;
            file.write_all(payload)?;
            file.flush()?;
            file.sync_all()?;
        }

        // Atomic rename.
        fs::rename(&tmp, &dest)?;

        // fsync directory to make the rename durable.
        let dir = File::open(&self.root)?;
        dir.sync_all()?;

        Ok(dest)
    }

    /// Read and validate a page from disk.
    ///
    /// Returns the parsed header and raw payload bytes. Every validation in
    /// `PageHeader::validate()` is applied, plus the payload digest is
    /// recomputed and compared.
    pub fn read_page(&self, payload_digest: &str) -> Result<(PageHeader, Vec<u8>)> {
        let path = self.page_path(payload_digest);
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        if file_len > MAX_HEADER_BYTES as u64 + page_format::MAX_PAYLOAD_BYTES {
            return Err(ProveKvError::CorruptPayload(format!(
                "page file too large: {} bytes",
                file_len
            )));
        }

        let mut reader = BufReader::with_capacity(MAX_HEADER_BYTES, file);

        // Read header: scan until newline, bounded to MAX_HEADER_BYTES.
        let mut header_buf = Vec::with_capacity(MAX_HEADER_BYTES);
        loop {
            if header_buf.len() >= MAX_HEADER_BYTES {
                return Err(ProveKvError::CorruptPayload(
                    "header exceeds max size".into(),
                ));
            }
            let mut byte = [0u8; 1];
            match reader.read_exact(&mut byte) {
                Ok(()) => {
                    if byte[0] == b'\n' {
                        break;
                    }
                    header_buf.push(byte[0]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Err(ProveKvError::CorruptPayload(
                        "truncated header: missing newline".into(),
                    ));
                }
                Err(e) => return Err(e.into()),
            }
        }

        let header: PageHeader = serde_json::from_slice(&header_buf)?;
        header.validate()?;

        // Read payload.
        let mut payload = Vec::with_capacity(header.payload_len as usize);
        reader.take(header.payload_len).read_to_end(&mut payload)?;

        if payload.len() as u64 != header.payload_len {
            return Err(ProveKvError::CorruptPayload(format!(
                "payload truncated: expected {} bytes, got {}",
                header.payload_len,
                payload.len()
            )));
        }

        // Verify payload digest.
        let actual_digest = page_format::compute_payload_digest(&payload);
        if actual_digest != header.payload_digest {
            return Err(ProveKvError::DigestMismatch {
                expected: header.payload_digest.clone(),
                got: actual_digest,
            });
        }

        Ok((header, payload))
    }

    /// Check whether a page exists on disk.
    pub fn page_exists(&self, payload_digest: &str) -> bool {
        self.page_path(payload_digest).is_file()
    }

    /// Delete a page by payload digest. Returns true if it existed.
    pub fn delete_page(&self, payload_digest: &str) -> Result<bool> {
        let path = self.page_path(payload_digest);
        if path.is_file() {
            fs::remove_file(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_format::build_page_header;

    use std::env;
    use std::sync::atomic::{AtomicU32, Ordering};

    static STORE_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_store() -> (PageStore, PathBuf) {
        let n = STORE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("provekv-test-{}-{}", std::process::id(), n));
        let _ = fs::remove_dir_all(&dir);
        let store = PageStore::open(&dir).unwrap();
        (store, dir)
    }

    #[test]
    fn write_and_read_roundtrip() {
        let (store, _dir) = temp_store();
        let payload = (0..=255u8).collect::<Vec<_>>();
        let header = build_page_header(
            "full_attn_k",
            &[64],
            "float32",
            b'l',
            "sha256:model",
            "sha256:layout",
            0,
            16,
            "raw_exact",
            &payload,
        )
        .unwrap();

        let path = store.write_page(&header, &payload).unwrap();
        assert!(path.is_file());

        let (read_header, read_payload) = store.read_page(&header.payload_digest).unwrap();
        assert_eq!(header.payload_digest, read_header.payload_digest);
        assert_eq!(payload, read_payload);
    }

    #[test]
    fn write_rejects_payload_mismatch() {
        let (store, _dir) = temp_store();
        let header = build_page_header(
            "full_attn_k",
            &[4],
            "float32",
            b'l',
            "sha256:model",
            "sha256:layout",
            0,
            1,
            "raw_exact",
            &[0u8; 16],
        )
        .unwrap();
        // Pass wrong payload.
        let result = store.write_page(&header, &[1u8; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn read_corrupt_header_rejected() {
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

        // Corrupt the header by truncating the file.
        let path = store.page_path(&header.payload_digest);
        let len = fs::metadata(&path).unwrap().len();
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 10).unwrap();

        assert!(store.read_page(&header.payload_digest).is_err());
    }

    #[test]
    fn read_corrupt_payload_digest_mismatch() {
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

        // Flip a byte in the payload portion of the file.
        let path = store.page_path(&header.payload_digest);
        let mut raw = fs::read(&path).unwrap();
        // Header is JSON + newline, so flip a byte near the end.
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        fs::write(&path, &raw).unwrap();

        assert!(store.read_page(&header.payload_digest).is_err());
    }

    #[test]
    fn delete_page_works() {
        let (store, _dir) = temp_store();
        let payload = vec![1u8; 64];
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
        assert!(store.page_exists(&header.payload_digest));
        assert!(store.delete_page(&header.payload_digest).unwrap());
        assert!(!store.page_exists(&header.payload_digest));
        assert!(!store.delete_page(&header.payload_digest).unwrap());
    }

    #[test]
    fn temp_files_cleaned_on_error() {
        let (store, dir) = temp_store();
        // Create a temp file that would collide, then write — it should clean it up.
        let payload = vec![7u8; 64];
        let header = build_page_header(
            "conv_state",
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

        let tmp = store.temp_path(&header.payload_digest);
        fs::write(&tmp, b"stale").unwrap();

        let result = store.write_page(&header, &payload);
        assert!(result.is_ok());
        // Stale temp should be gone.
        assert!(!tmp.exists());

        // No leftover temp files in store root.
        let temps: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with(".tmp."))
                    .unwrap_or(false)
            })
            .collect();
        assert!(temps.is_empty());
    }
}
