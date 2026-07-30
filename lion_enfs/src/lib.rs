// lion_enfs/src/lib.rs — ENFS Public API
//
// Einstein Neurons File System — AI-native binary filesystem.
//
// All data stored in ENFS is opaque binary. No human-readable names,
// no human-readable values, no plaintext on disk.
// Only AI systems with access to the BLAKE3-keyed index can resolve names.

pub mod block;
pub mod error;
pub mod header;
pub mod index;
pub mod inode;
pub mod manifest;
pub mod memory;
pub mod storage;
pub mod volume;

pub use block::{BlockAllocator, BlockCodec, BlockKind, BlockHeader};
pub use error::{EnfsError, EnfsResult};
pub use header::{EnfsHeader, StorageTier, VolumeFlags, ENFS_MAGIC};
pub use index::{TensorIndex, DomainTag, IndexEntry};
pub use inode::{Inode, InodeTable, InodeKind, TensorDtype};
pub use manifest::{EnfsManifest, Modality, QuantizationKind};
pub use memory::{MemoryHierarchy, SensoryMemory, WorkingMemory, DomainMemory, DomainKey, ArchiveMemory};
pub use volume::Volume;

/// ENFS crate version
pub const ENFS_CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
