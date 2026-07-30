// lion_enfs/src/block.rs — ENFS Block Allocator & Block Types
//
// A "block" is the smallest unit of disk allocation in ENFS.
// Block size is chosen per storage tier (4KiB for NVMe/RAM, up to 512KiB for HDD).
// Every block carries its own BLAKE3 checksum and compression codec byte.

use serde::{Deserialize, Serialize};
use crate::error::{EnfsError, EnfsResult};

/// Compression codec applied to a single block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlockCodec {
    None  = 0,
    Lz4   = 1,
    Zstd  = 2,
}

/// Block type — what kind of ENFS data this block holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlockKind {
    Free        = 0,
    Header      = 1,
    Manifest    = 2,
    Inode       = 3,
    TensorData  = 4,
    IndexEntry  = 5,
    DomainData  = 6,
    AdapterData = 7,
    CacheData   = 8,
    Checkpoint  = 9,
    Signature   = 10,
    Custom      = 255,
}

/// Fixed per-block header (32 bytes) prepended to every block's data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    pub block_id:        u64,
    pub kind:            BlockKind,
    pub codec:           BlockCodec,
    pub payload_len:     u32,          // compressed length
    pub original_len:    u32,          // uncompressed length
    pub blake3:          [u8; 16],     // first 16 bytes of BLAKE3(payload)
    pub _pad:            [u8; 6],
}

impl BlockHeader {
    pub const SIZE: usize = 32;

    pub fn new(block_id: u64, kind: BlockKind, codec: BlockCodec,
               payload: &[u8], original_len: u32) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(payload);
        let hash = hasher.finalize();
        let mut blake3 = [0u8; 16];
        blake3.copy_from_slice(&hash.as_bytes()[..16]);

        Self {
            block_id,
            kind,
            codec,
            payload_len: payload.len() as u32,
            original_len,
            blake3,
            _pad: [0u8; 6],
        }
    }

    pub fn verify(&self, payload: &[u8]) -> EnfsResult<()> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(payload);
        let hash = hasher.finalize();
        let computed = &hash.as_bytes()[..16];
        if computed != self.blake3 {
            return Err(EnfsError::IntegrityFailure {
                reason: format!("Block {} BLAKE3 mismatch", self.block_id),
            });
        }
        Ok(())
    }
}

/// Block allocator: tracks which block IDs are free.
/// Uses a compact bitfield internally for O(1) free-block lookup.
#[derive(Debug, Default)]
pub struct BlockAllocator {
    total_blocks: u64,
    free_bitmap:  Vec<u64>,   // 1 bit per block, 64 blocks per u64
    next_free_hint: u64,
}

impl BlockAllocator {
    pub fn new(total_blocks: u64) -> Self {
        let words = ((total_blocks + 63) / 64) as usize;
        // All bits set = all blocks free
        let mut free_bitmap = vec![u64::MAX; words];
        // Mark header region (first 384 KiB worth of blocks) as allocated
        // Header(128B) + InodeTable(64KiB) + Manifest(64KiB) + Index(256KiB) = 384KiB
        let reserved = (384 * 1024 / 4096).min(total_blocks) as usize;
        for i in 0..reserved {
            let word = i / 64;
            let bit  = i % 64;
            free_bitmap[word] &= !(1u64 << bit);
        }
        Self { total_blocks, free_bitmap, next_free_hint: reserved as u64 }
    }

    /// Allocate the next free block. Returns block ID.
    pub fn alloc(&mut self) -> EnfsResult<u64> {
        let start_word = (self.next_free_hint / 64) as usize;
        let n_words = self.free_bitmap.len();
        for offset in 0..n_words {
            let idx = (start_word + offset) % n_words;
            let word = self.free_bitmap[idx];
            if word != 0 {
                let bit = word.trailing_zeros() as u64;
                let block_id = idx as u64 * 64 + bit;
                if block_id < self.total_blocks {
                    self.free_bitmap[idx] &= !(1u64 << bit);
                    self.next_free_hint = block_id + 1;
                    return Ok(block_id);
                }
            }
        }
        Err(EnfsError::VolumeFull)
    }

    /// Free a previously allocated block.
    pub fn free(&mut self, block_id: u64) {
        if block_id < self.total_blocks {
            let word = (block_id / 64) as usize;
            let bit  = block_id % 64;
            self.free_bitmap[word] |= 1u64 << bit;
            if block_id < self.next_free_hint {
                self.next_free_hint = block_id;
            }
        }
    }

    pub fn free_count(&self) -> u64 {
        self.free_bitmap.iter().map(|w| w.count_ones() as u64).sum()
    }

    pub fn used_count(&self) -> u64 {
        self.total_blocks - self.free_count()
    }
}

// ── Compression helpers ────────────────────────────────────────────────────────

pub fn compress(codec: BlockCodec, data: &[u8]) -> EnfsResult<Vec<u8>> {
    match codec {
        BlockCodec::None => Ok(data.to_vec()),
        BlockCodec::Lz4  => {
            lz4_flex::compress_prepend_size(data)
                .pipe_ok()
        }
        BlockCodec::Zstd => {
            zstd::encode_all(data, 3)
                .map_err(|e| EnfsError::CompressionError(e.to_string()))
        }
    }
}

pub fn decompress(codec: BlockCodec, data: &[u8]) -> EnfsResult<Vec<u8>> {
    match codec {
        BlockCodec::None => Ok(data.to_vec()),
        BlockCodec::Lz4  => {
            lz4_flex::decompress_size_prepended(data)
                .map_err(|e| EnfsError::CompressionError(e.to_string()))
        }
        BlockCodec::Zstd => {
            zstd::decode_all(data)
                .map_err(|e| EnfsError::CompressionError(e.to_string()))
        }
    }
}

// Helper trait to unify lz4 result
trait PipeOk<T> {
    fn pipe_ok(self) -> EnfsResult<T>;
}
impl PipeOk<Vec<u8>> for Vec<u8> {
    fn pipe_ok(self) -> EnfsResult<Vec<u8>> { Ok(self) }
}
