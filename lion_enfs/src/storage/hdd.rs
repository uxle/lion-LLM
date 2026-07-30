// lion_enfs/src/storage/hdd.rs — HDD Backend
//
// Strategy: large sequential reads (512 KiB blocks) to amortise seek latency.
// Queue depth: 1 (HDDs have a single mechanical head — deep queuing hurts).
// Writes are buffered and flushed sequentially.
// Target: 100–200 MB/s sequential.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Mutex;
use crate::error::{EnfsError, EnfsResult};
use crate::header::StorageTier;
use super::StorageBackend;

pub struct HddBackend {
    path: PathBuf,
    file: Mutex<File>,
}

impl HddBackend {
    pub async fn open(path: &Path) -> EnfsResult<Self> {
        let volume_file = path.join("enfs.vol");
        let std_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&volume_file)
            .map_err(EnfsError::Io)?;

        // Pre-allocate 256 MiB if new
        if std_file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            std_file.set_len(256 * 1024 * 1024).map_err(EnfsError::Io)?;
        }

        let file = File::from_std(std_file);
        Ok(Self { path: path.to_path_buf(), file: Mutex::new(file) })
    }
}

#[async_trait]
impl StorageBackend for HddBackend {
    fn tier(&self) -> StorageTier { StorageTier::Hdd }
    fn throughput_mbps(&self) -> u32 { StorageTier::Hdd.throughput_mbps() }
    fn block_size(&self) -> u32 { 512 * 1024 } // 512 KiB — amortises seek cost
    fn queue_depth(&self) -> u32 { 1 }         // single mechanical head

    async fn read_at(&self, offset: u64, len: usize) -> EnfsResult<Vec<u8>> {
        let mut file = self.file.lock().await;
        file.seek(SeekFrom::Start(offset)).await.map_err(EnfsError::Io)?;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf).await.map_err(EnfsError::Io)?;
        Ok(buf)
    }

    async fn write_at(&self, offset: u64, data: &[u8]) -> EnfsResult<()> {
        let mut file = self.file.lock().await;
        file.seek(SeekFrom::Start(offset)).await.map_err(EnfsError::Io)?;
        file.write_all(data).await.map_err(EnfsError::Io)?;
        Ok(())
    }

    async fn flush(&self) -> EnfsResult<()> {
        let mut file = self.file.lock().await;
        file.flush().await.map_err(EnfsError::Io)
    }

    async fn trim(&self, _offset: u64, _len: usize) -> EnfsResult<()> {
        // HDDs don't support TRIM
        Ok(())
    }
}
