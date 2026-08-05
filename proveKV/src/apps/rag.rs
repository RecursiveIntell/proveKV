//! RAG document KV-cache capture and composition.
//!
//! The adapter deliberately treats a document cache as an opaque, ordered
//! sequence of KV values.  The document digest is computed from the source
//! bytes (not from the cache), so a cache can be looked up before decoding or
//! composing it.

use serde::{Deserialize, Serialize};

/// A KV cache captured for one document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentCache {
    /// BLAKE3 digest of the document content, encoded as lowercase hex.
    pub document_hash: String,
    /// Opaque flattened KV values in document order.
    pub kv_cache: Vec<f32>,
}

/// A composed RAG context containing the caches of several documents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RAGContext {
    /// Individual document caches, retained for provenance and lookup.
    pub documents: Vec<DocumentCache>,
    /// The grafted cache, formed by concatenating document caches in order.
    pub kv_cache: Vec<f32>,
}

impl DocumentCache {
    /// Capture `kv_cache` and content-address it by `document`.
    pub fn capture(document: impl AsRef<[u8]>, kv_cache: Vec<f32>) -> Self {
        Self {
            document_hash: blake3::hash(document.as_ref()).to_hex().to_string(),
            kv_cache,
        }
    }

    /// Return whether this cache belongs to `document`.
    pub fn matches(&self, document: impl AsRef<[u8]>) -> bool {
        self.document_hash == blake3::hash(document.as_ref()).to_hex().to_string()
    }
}

/// Capture a document's already-produced KV cache.
pub fn capture(document: impl AsRef<[u8]>, kv_cache: Vec<f32>) -> DocumentCache {
    DocumentCache::capture(document, kv_cache)
}

/// Compose multiple document caches into one ordered RAG context.
pub fn graft<'a, I>(documents: I) -> RAGContext
where
    I: IntoIterator<Item = &'a DocumentCache>,
{
    let documents: Vec<DocumentCache> = documents.into_iter().cloned().collect();
    let total = documents.iter().map(|d| d.kv_cache.len()).sum();
    let mut kv_cache = Vec::with_capacity(total);
    for document in &documents {
        kv_cache.extend_from_slice(&document.kv_cache);
    }
    RAGContext {
        documents,
        kv_cache,
    }
}

impl RAGContext {
    /// Compose document caches in the supplied order.
    pub fn graft<'a, I>(documents: I) -> Self
    where
        I: IntoIterator<Item = &'a DocumentCache>,
    {
        graft(documents)
    }

    /// Find a captured document by its source content.
    pub fn find(&self, document: impl AsRef<[u8]>) -> Option<&DocumentCache> {
        self.documents
            .iter()
            .find(|cache| cache.matches(document.as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_hashes_source_content() {
        let cache = capture("document one", vec![1.0, 2.0]);
        assert_eq!(
            cache.document_hash,
            blake3::hash(b"document one").to_hex().to_string()
        );
        assert!(cache.matches("document one"));
        assert!(!cache.matches("document two"));
    }

    #[test]
    fn graft_preserves_document_and_cache_order() {
        let first = capture("first", vec![1.0, 2.0]);
        let second = capture("second", vec![3.0]);
        let context = graft([&first, &second]);
        assert_eq!(context.documents, vec![first, second]);
        assert_eq!(context.kv_cache, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn graft_empty_is_empty() {
        let context = RAGContext::graft(std::iter::empty());
        assert!(context.documents.is_empty());
        assert!(context.kv_cache.is_empty());
    }
}
