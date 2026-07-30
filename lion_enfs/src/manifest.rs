// lion_enfs/src/manifest.rs — ENFS Volume Manifest
//
// The manifest is stored in binary (bincode) at a fixed offset after the header.
// It describes model identity, modalities, quantizations, and runtime requirements.
// No human-readable fields are stored in the binary region.
// An optional JSON metadata blob is embedded but sealed inside the binary block.

use serde::{Deserialize, Serialize};
use crate::error::{EnfsError, EnfsResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Modality {
    Text       = 0,
    Image      = 1,
    Audio      = 2,
    Video      = 3,
    Mesh3d     = 4,
    PointCloud = 5,
    DepthMap   = 6,
    TimeSeries = 7,
    Sensor     = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum QuantizationKind {
    F32  = 0,
    Bf16 = 1,
    F16  = 2,
    F8   = 3,
    Int8 = 4,
    Int4 = 5,
    Nf4  = 6,
    Q4   = 7,
    Q8   = 8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnfsManifest {
    // Identity — stored as BLAKE3 hashes to keep the format opaque
    pub model_name_hash:    [u8; 32],
    pub author_hash:        [u8; 32],
    pub license_crc32:      u32,         // compact checksum of SPDX license string
    pub created_at:         u64,
    pub updated_at:         u64,

    // Capability flags
    pub modalities:         Vec<Modality>,
    pub quantizations:      Vec<QuantizationKind>,

    // Model dimensions (always numeric, never string)
    pub context_length:     u64,
    pub embedding_size:     u64,
    pub hidden_size:        u64,
    pub layer_count:        u32,
    pub head_count:         u32,
    pub kv_heads:           u32,
    pub vocab_size:         u32,
    pub parameter_count:    u64,
    pub training_tokens:    u64,

    // Hardware hints (in MiB)
    pub recommended_ram_mb:  u64,
    pub recommended_vram_mb: u64,

    // Integrity
    pub manifest_blake3:    [u8; 32],   // BLAKE3 of all fields above (set last)
}

impl EnfsManifest {
    pub fn new(
        model_name: &str,
        author: &str,
        modalities: Vec<Modality>,
        quantizations: Vec<QuantizationKind>,
    ) -> Self {
        let now = unix_now();
        Self {
            model_name_hash:    blake3_of(model_name.as_bytes()),
            author_hash:        blake3_of(author.as_bytes()),
            license_crc32:      0,
            created_at:         now,
            updated_at:         now,
            modalities,
            quantizations,
            context_length:     0,
            embedding_size:     0,
            hidden_size:        0,
            layer_count:        0,
            head_count:         0,
            kv_heads:           0,
            vocab_size:         0,
            parameter_count:    0,
            training_tokens:    0,
            recommended_ram_mb:  0,
            recommended_vram_mb: 0,
            manifest_blake3:    [0u8; 32],
        }
    }

    /// Compute and seal the manifest's own BLAKE3 hash.
    pub fn seal(&mut self) -> EnfsResult<()> {
        let mut tmp = self.clone();
        tmp.manifest_blake3 = [0u8; 32];
        let bytes = bincode::serialize(&tmp)
            .map_err(|e| EnfsError::Serialization(e.to_string()))?;
        self.manifest_blake3 = blake3_of(&bytes);
        Ok(())
    }

    pub fn to_bytes(&self) -> EnfsResult<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| EnfsError::Serialization(e.to_string()))
    }

    pub fn from_bytes(buf: &[u8]) -> EnfsResult<Self> {
        bincode::deserialize(buf)
            .map_err(|e| EnfsError::Serialization(e.to_string()))
    }
}

fn blake3_of(data: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(data);
    *h.finalize().as_bytes()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
