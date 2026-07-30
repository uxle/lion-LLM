// lion_enfs/src/storage/mod.rs — Storage Backend Trait & Auto-Detection
//
// ENFS adapts its I/O strategy based on detected storage tier.
// Each backend implements the same `StorageBackend` trait.

pub mod detect;
pub mod hdd;
pub mod ssd;
pub mod nvme;

use std::path::Path;
use async_trait::async_trait;
use crate::error::EnfsResult;
use crate::header::StorageTier;

// ── Backend Trait ──────────────────────────────────────────────────────────────

#[async_trait]
pub trait StorageBackend: Send + Sync {
    fn tier(&self) -> StorageTier;
    fn throughput_mbps(&self) -> u32;
    fn block_size(&self) -> u32;
    fn queue_depth(&self) -> u32;

    /// Read exactly `len` bytes starting at `offset`.
    async fn read_at(&self, offset: u64, len: usize) -> EnfsResult<Vec<u8>>;

    /// Write `data` at `offset`. Must be block-aligned.
    async fn write_at(&self, offset: u64, data: &[u8]) -> EnfsResult<()>;

    /// Flush any write buffers to durable storage.
    async fn flush(&self) -> EnfsResult<()>;

    /// Punch a hole (TRIM/discard) for freed blocks when supported.
    async fn trim(&self, offset: u64, len: usize) -> EnfsResult<()>;

    fn supports_mmap(&self) -> bool { false }
    fn supports_direct_io(&self) -> bool { false }
}

// ── Backend Selection ─────────────────────────────────────────────────────────

/// Select the best available backend for the given path, or honour an explicit override.
pub async fn open_backend(
    path: &Path,
    override_tier: Option<StorageTier>,
) -> EnfsResult<Box<dyn StorageBackend>> {
    let tier = match override_tier {
        Some(t) if t != StorageTier::Auto => t,
        _ => detect::detect_storage_tier(path),
    };

    match tier {
        StorageTier::NVMe | StorageTier::Ram => {
            Ok(Box::new(nvme::NvmeBackend::open(path).await?))
        }
        StorageTier::Ssd => {
            Ok(Box::new(ssd::SsdBackend::open(path).await?))
        }
        StorageTier::Hdd | StorageTier::Auto => {
            Ok(Box::new(hdd::HddBackend::open(path).await?))
        }
    }
}
