// lion_enfs/src/storage/nvme.rs — NVMe / RAM Backend
//
// Strategy: memory-mapped I/O via memmap2 + direct buffered writes.
// Queue depth: 1024 (NVMe), 4096 (RAM-backed).
// Block size: 4 KiB (page-aligned).
// Target throughput: 3,000 – 25,000+ MB/s.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use memmap2::MmapMut;
use crate::error::{EnfsError, EnfsResult};
use crate::header::StorageTier;
use super::StorageBackend;

pub struct NvmeBackend {
    path:       PathBuf,
    tier:       StorageTier,
    mmap:       Arc<RwLock<MmapMut>>,
    file_len:   u64,
}

impl NvmeBackend {
    pub async fn open(path: &Path) -> EnfsResult<Self> {
        // Determine if this is RAM-backed
        let tier = super::detect::detect_storage_tier(path);
        let actual_tier = if tier == StorageTier::Ram { StorageTier::Ram } else { StorageTier::NVMe };

        // Open the volume file (create if absent, pre-allocate)
        let volume_file = path.join("enfs.vol");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&volume_file)
            .map_err(EnfsError::Io)?;

        // Pre-allocate if new (default 1 GiB; real volumes sized at creation)
        let default_size: u64 = 1 * 1024 * 1024 * 1024;
        let file_len = match file.metadata() {
            Ok(m) if m.len() > 0 => m.len(),
            _ => {
                file.set_len(default_size).map_err(EnfsError::Io)?;
                default_size
            }
        };

        let mmap = unsafe { MmapMut::map_mut(&file).map_err(EnfsError::Io)? };

        Ok(Self {
            path: path.to_path_buf(),
            tier: actual_tier,
            mmap: Arc::new(RwLock::new(mmap)),
            file_len,
        })
    }
}

#[async_trait]
impl StorageBackend for NvmeBackend {
    fn tier(&self) -> StorageTier { self.tier }
    fn throughput_mbps(&self) -> u32 { self.tier.throughput_mbps() }
    fn block_size(&self) -> u32 { 4_096 }
    fn queue_depth(&self) -> u32 { self.tier.queue_depth() }
    fn supports_mmap(&self) -> bool { true }
    fn supports_direct_io(&self) -> bool { true }

    async fn read_at(&self, offset: u64, len: usize) -> EnfsResult<Vec<u8>> {
        let mmap = self.mmap.read().await;
        let end = (offset as usize) + len;
        if end > mmap.len() {
            return Err(EnfsError::StorageError(format!(
                "NVMe read out of bounds: offset={} len={} file_len={}",
                offset, len, mmap.len()
            )));
        }
        Ok(mmap[offset as usize..end].to_vec())
    }

    async fn write_at(&self, offset: u64, data: &[u8]) -> EnfsResult<()> {
        let mut mmap = self.mmap.write().await;
        let end = (offset as usize) + data.len();
        if end > mmap.len() {
            return Err(EnfsError::StorageError(format!(
                "NVMe write out of bounds: offset={} len={}", offset, data.len()
            )));
        }
        mmap[offset as usize..end].copy_from_slice(data);
        Ok(())
    }

    async fn flush(&self) -> EnfsResult<()> {
        let mmap = self.mmap.read().await;
        mmap.flush().map_err(EnfsError::Io)
    }

    async fn trim(&self, _offset: u64, _len: usize) -> EnfsResult<()> {
        // mmap-backed: no-op (page cache handles reclamation)
        Ok(())
    }
}
