// lion_enfs/src/header.rs — ENFS Volume Header
//
// Binary layout (fixed 128 bytes):
//
//   Offset  Size  Field
//   0       4     Magic: 0x454E4653 ("ENFS")
//   4       2     version_major
//   6       2     version_minor
//   8       2     version_patch
//   10      1     endianness (0=LE, 1=BE)
//   11      1     flags
//   12      4     block_size_bytes (default 4096)
//   16      8     volume_size_blocks
//   24      8     inode_table_offset
//   32      8     manifest_offset
//   40      8     index_offset
//   48      8     data_region_offset
//   56      8     created_at_unix_secs
//   64      8     modified_at_unix_secs
//   72      4     storage_tier (0=HDD, 1=SSD, 2=NVMe, 3=RAM, 255=Auto)
//   76      4     max_throughput_mbps
//   80      32    volume_uuid (16 bytes) + reserved (16 bytes)
//   112     16    root_blake3_partial (first 16 bytes of root hash)
//   — total: 128 bytes

use serde::{Deserialize, Serialize};
use crate::error::{EnfsError, EnfsResult};

/// ENFS magic bytes: "ENFS" in ASCII
pub const ENFS_MAGIC: [u8; 4] = [0x45, 0x4E, 0x46, 0x53];

pub const ENFS_VERSION_MAJOR: u16 = 1;
pub const ENFS_VERSION_MINOR: u16 = 0;
pub const ENFS_VERSION_PATCH: u16 = 0;

/// Default block size: 4 KiB (page-aligned, NVMe optimal)
pub const DEFAULT_BLOCK_SIZE: u32 = 4096;

/// Header is always exactly 128 bytes
pub const HEADER_SIZE: usize = 128;

/// Storage tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum StorageTier {
    Hdd   = 0,   // ~100–200 MB/s sequential
    Ssd   = 1,   // ~500–600 MB/s sequential
    NVMe  = 2,   // ~3,000–7,000 MB/s sequential
    Ram   = 3,   // ~25,000+ MB/s
    Auto  = 255, // Detected at runtime
}

impl StorageTier {
    /// Theoretical peak sequential read throughput in MB/s (conservative floor estimate).
    pub fn throughput_mbps(&self) -> u32 {
        match self {
            StorageTier::Hdd  => 180,
            StorageTier::Ssd  => 560,
            StorageTier::NVMe => 7_000,
            StorageTier::Ram  => 25_000,
            StorageTier::Auto => 0, // filled after detection
        }
    }

    /// Recommended I/O queue depth for this tier.
    pub fn queue_depth(&self) -> u32 {
        match self {
            StorageTier::Hdd  => 1,
            StorageTier::Ssd  => 32,
            StorageTier::NVMe => 1024,
            StorageTier::Ram  => 4096,
            StorageTier::Auto => 32,
        }
    }

    /// Recommended block size for aligned reads on this tier.
    pub fn optimal_block_size(&self) -> u32 {
        match self {
            StorageTier::Hdd  => 512 * 1024,  // 512 KiB — sequential reads amortize seek cost
            StorageTier::Ssd  => 128 * 1024,  // 128 KiB
            StorageTier::NVMe => 4_096,        // 4 KiB — NVMe page size
            StorageTier::Ram  => 4_096,        // 4 KiB — CPU page size
            StorageTier::Auto => 4_096,
        }
    }
}

bitflags::bitflags! {
    /// Volume feature flags — stored on disk as a plain u8.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct VolumeFlags: u8 {
        /// Volume is read-only
        const READ_ONLY      = 0b0000_0001;
        /// Per-block encryption enabled
        const ENCRYPTED      = 0b0000_0010;
        /// Per-block compression enabled
        const COMPRESSED     = 0b0000_0100;
        /// Human-readable access denied (always set for ENFS)
        const AI_ONLY        = 0b0000_1000;
        /// Volume supports delta updates
        const DELTA_UPDATES  = 0b0001_0000;
        /// Volume is a RAM-backed ephemeral volume
        const EPHEMERAL      = 0b0010_0000;
    }
}

impl Default for VolumeFlags {
    fn default() -> Self {
        VolumeFlags::AI_ONLY
    }
}

impl VolumeFlags {
    pub fn to_u8(self) -> u8 { self.bits() }
    pub fn from_u8(v: u8) -> Self { Self::from_bits_truncate(v) }
}

/// ENFS Volume Header — fixed 128-byte structure at offset 0 of every volume.
/// `flags` is stored as a plain `u8` for serde/bincode compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnfsHeader {
    pub magic: [u8; 4],
    pub version_major: u16,
    pub version_minor: u16,
    pub version_patch: u16,
    pub endianness: u8,         // 0 = little-endian
    pub flags: u8,              // VolumeFlags bits — use flags() / set_flags() helpers
    pub block_size_bytes: u32,
    pub volume_size_blocks: u64,
    pub inode_table_offset: u64,
    pub manifest_offset: u64,
    pub index_offset: u64,
    pub data_region_offset: u64,
    pub created_at_unix_secs: u64,
    pub modified_at_unix_secs: u64,
    pub storage_tier: u32,      // StorageTier as u32
    pub max_throughput_mbps: u32,
    pub volume_uuid: [u8; 16],
    pub _reserved: [u8; 16],
    pub root_blake3_partial: [u8; 16], // first 16 bytes of full volume BLAKE3
}

impl EnfsHeader {
    pub fn new(
        volume_size_blocks: u64,
        storage_tier: StorageTier,
        volume_uuid: [u8; 16],
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let block_size = storage_tier.optimal_block_size();

        // Fixed offsets after 128-byte header
        let inode_table_offset  = HEADER_SIZE as u64;
        let manifest_offset     = inode_table_offset + (64 * 1024);    // 64 KiB inode table
        let index_offset        = manifest_offset    + (64 * 1024);    // 64 KiB manifest
        let data_region_offset  = index_offset       + (256 * 1024);   // 256 KiB index

        Self {
            magic: ENFS_MAGIC,
            version_major: ENFS_VERSION_MAJOR,
            version_minor: ENFS_VERSION_MINOR,
            version_patch: ENFS_VERSION_PATCH,
            endianness: 0,
            flags: VolumeFlags::AI_ONLY.to_u8(),
            block_size_bytes: block_size,
            volume_size_blocks,
            inode_table_offset,
            manifest_offset,
            index_offset,
            data_region_offset,
            created_at_unix_secs: now,
            modified_at_unix_secs: now,
            storage_tier: storage_tier as u32,
            max_throughput_mbps: storage_tier.throughput_mbps(),
            volume_uuid,
            _reserved: [0u8; 16],
            root_blake3_partial: [0u8; 16],
        }
    }

    /// Validate magic bytes and version compatibility.
    pub fn validate(&self) -> EnfsResult<()> {
        if self.magic != ENFS_MAGIC {
            return Err(EnfsError::InvalidMagic);
        }
        if self.version_major != ENFS_VERSION_MAJOR {
            return Err(EnfsError::UnsupportedVersion {
                major: self.version_major,
                minor: self.version_minor,
            });
        }
        Ok(())
    }

    /// Serialize to fixed 128-byte binary buffer.
    pub fn to_bytes(&self) -> EnfsResult<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| EnfsError::Serialization(e.to_string()))
    }

    /// Deserialize from binary buffer and validate.
    pub fn from_bytes(buf: &[u8]) -> EnfsResult<Self> {
        let h: Self = bincode::deserialize(buf)
            .map_err(|e| EnfsError::Serialization(e.to_string()))?;
        h.validate()?;
        Ok(h)
    }

    pub fn volume_flags(&self) -> VolumeFlags {
        VolumeFlags::from_u8(self.flags)
    }

    pub fn set_volume_flags(&mut self, f: VolumeFlags) {
        self.flags = f.to_u8();
    }

    pub fn storage_tier_enum(&self) -> StorageTier {
        match self.storage_tier {
            0   => StorageTier::Hdd,
            1   => StorageTier::Ssd,
            2   => StorageTier::NVMe,
            3   => StorageTier::Ram,
            _   => StorageTier::Auto,
        }
    }
}
