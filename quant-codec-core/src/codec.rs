use crate::{CodecId, CodecProfileDigest, KvSliceRequest, KvTensorShape};

pub trait CodecProfile {
    fn codec_id(&self) -> CodecId;
    fn codec_version(&self) -> &str;
    fn profile_digest(&self) -> CodecProfileDigest;
    fn fixed_rate_bits(&self) -> Option<u16>;
    fn block_dim(&self) -> Option<u16>;
    fn is_lossy(&self) -> bool;
}

pub trait VectorCodec {
    type EncodedBlock;
    type Error;

    fn encode_block(&self, input: &[f32]) -> Result<Self::EncodedBlock, Self::Error>;
    fn decode_block(&self, block: &Self::EncodedBlock, out: &mut [f32]) -> Result<(), Self::Error>;
}

pub trait KvCacheCodec: VectorCodec {
    type EncodedCache;

    fn encode_kv_cache(
        &self,
        tensors: &[f32],
        shape: KvTensorShape,
    ) -> Result<Self::EncodedCache, Self::Error>;

    fn decode_slice(
        &self,
        cache: &Self::EncodedCache,
        request: KvSliceRequest,
        out: &mut [f32],
    ) -> Result<(), Self::Error>;
}
