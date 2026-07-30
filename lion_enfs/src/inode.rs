// lion_enfs/src/inode.rs — ENFS Inode Table
//
// An inode describes one logical entity in ENFS (a tensor, domain store, adapter pack, etc.).
// Every inode has a stable numeric ID, a Blake3-verified name hash, type metadata,
// and a list of block IDs that store its payload.
//
// Inodes are NEVER stored with human-readable names on disk.
// Only a BLAKE3 hash of the name is persisted; the index maps name → inode ID in RAM.

use serde::{Deserialize, Serialize};
use crate::error::{EnfsError, EnfsResult};

/// What kind of object this inode represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum InodeKind {
    Directory      = 0,
    Tensor         = 1,
    DomainStore    = 2,
    Tokenizer      = 3,
    Adapter        = 4,
    Checkpoint     = 5,
    CacheEntry     = 6,
    Signature      = 7,
    Plugin         = 8,
    Manifest       = 9,
    IndexEntry     = 10,
}

/// Tensor data type stored in a Tensor inode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TensorDtype {
    F64  = 0,
    F32  = 1,
    Bf16 = 2,
    F16  = 3,
    F8   = 4,
    I64  = 10,
    I32  = 11,
    I16  = 12,
    I8   = 13,
    I4   = 14,
    Nf4  = 20,
    Q4   = 21,
    Q8   = 22,
}

/// Fixed 64-byte inode record stored in the inode table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inode {
    /// Stable numeric ID (never reused within a volume lifetime).
    pub inode_id:       u64,
    /// BLAKE3 hash of the canonical name — not the name itself.
    pub name_hash:      [u8; 32],
    pub kind:           InodeKind,
    pub dtype:          TensorDtype,    // only meaningful for Tensor inodes
    /// Shape of tensor (up to 8 dims). Zero-padded.
    pub shape:          [u64; 8],
    pub ndim:           u8,
    pub size_bytes:     u64,            // total uncompressed payload bytes
    pub block_count:    u32,
    /// Offset of the block-ID list in the index region.
    pub block_list_offset: u64,
    pub created_at:     u64,
    pub modified_at:    u64,
    /// BLAKE3 of full payload (all blocks concatenated, decompressed).
    pub payload_blake3: [u8; 32],
    pub flags:          u8,
    pub _pad:           [u8; 7],
}

impl Inode {
    pub const SIZE: usize = 64 + 32 + 32 + 8*8 + 8 + 8 + 4 + 8 + 8 + 8 + 32 + 1 + 7;
    // ~232 bytes; stored in 256-byte slots for alignment.

    pub fn new_tensor(
        inode_id: u64,
        name: &str,
        dtype: TensorDtype,
        shape: &[u64],
        size_bytes: u64,
    ) -> Self {
        let name_hash = blake3_of(name.as_bytes());
        let mut shape_arr = [0u64; 8];
        let ndim = shape.len().min(8);
        shape_arr[..ndim].copy_from_slice(&shape[..ndim]);

        let now = unix_now();
        Self {
            inode_id,
            name_hash,
            kind: InodeKind::Tensor,
            dtype,
            shape: shape_arr,
            ndim: ndim as u8,
            size_bytes,
            block_count: 0,
            block_list_offset: 0,
            created_at: now,
            modified_at: now,
            payload_blake3: [0u8; 32],
            flags: 0,
            _pad: [0u8; 7],
        }
    }

    pub fn new_directory(inode_id: u64, name: &str) -> Self {
        let name_hash = blake3_of(name.as_bytes());
        let now = unix_now();
        Self {
            inode_id,
            name_hash,
            kind: InodeKind::Directory,
            dtype: TensorDtype::F32,
            shape: [0u64; 8],
            ndim: 0,
            size_bytes: 0,
            block_count: 0,
            block_list_offset: 0,
            created_at: now,
            modified_at: now,
            payload_blake3: [0u8; 32],
            flags: 0,
            _pad: [0u8; 7],
        }
    }

    /// Finalize payload hash after all blocks are written.
    pub fn finalize_hash(&mut self, payload: &[u8]) {
        self.payload_blake3 = blake3_of(payload);
        self.modified_at = unix_now();
    }

    /// Verify payload against stored hash.
    pub fn verify_payload(&self, payload: &[u8]) -> EnfsResult<()> {
        let computed = blake3_of(payload);
        if computed != self.payload_blake3 {
            return Err(EnfsError::IntegrityFailure {
                reason: format!("Inode {} payload BLAKE3 mismatch", self.inode_id),
            });
        }
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

/// In-memory inode table: maps inode_id → Inode.
/// Backed by a flat Vec for cache-locality; rarely >64K inodes in practice.
#[derive(Debug, Default)]
pub struct InodeTable {
    inodes:    Vec<Option<Inode>>,
    next_id:   u64,
}

impl InodeTable {
    pub fn new() -> Self {
        Self { inodes: Vec::new(), next_id: 1 }
    }

    pub fn allocate(&mut self, inode: Inode) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        if id as usize >= self.inodes.len() {
            self.inodes.resize_with(id as usize + 1, || None);
        }
        self.inodes[id as usize] = Some(inode);
        id
    }

    pub fn get(&self, id: u64) -> EnfsResult<&Inode> {
        self.inodes
            .get(id as usize)
            .and_then(|s| s.as_ref())
            .ok_or(EnfsError::InodeNotFound { inode_id: id })
    }

    pub fn get_mut(&mut self, id: u64) -> EnfsResult<&mut Inode> {
        self.inodes
            .get_mut(id as usize)
            .and_then(|s| s.as_mut())
            .ok_or(EnfsError::InodeNotFound { inode_id: id })
    }

    pub fn free(&mut self, id: u64) {
        if let Some(slot) = self.inodes.get_mut(id as usize) {
            *slot = None;
        }
    }

    pub fn count(&self) -> usize {
        self.inodes.iter().filter(|s| s.is_some()).count()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn blake3_of(data: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
