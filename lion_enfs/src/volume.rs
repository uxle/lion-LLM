// lion_enfs/src/volume.rs — ENFS Volume (Top-Level API)
//
// A Volume is the main object an AI system interacts with.
// It owns: header, manifest, inode table, index, block allocator, memory hierarchy, and backend.
//
// All data is stored in opaque binary. No human-readable path names exist on disk.

use std::path::{Path, PathBuf};

use crate::error::{EnfsError, EnfsResult};
use crate::header::{EnfsHeader, StorageTier};
use crate::manifest::{EnfsManifest, Modality, QuantizationKind};
use crate::inode::{Inode, InodeTable, TensorDtype};
use crate::index::{TensorIndex, DomainTag};
use crate::block::{BlockAllocator, BlockCodec, BlockKind, BlockHeader, compress, decompress};
use crate::memory::MemoryHierarchy;
use crate::storage::{StorageBackend, open_backend};

pub struct Volume {
    pub header:    EnfsHeader,
    pub manifest:  EnfsManifest,
    inodes:        InodeTable,
    index:         TensorIndex,
    allocator:     BlockAllocator,
    memory:        MemoryHierarchy,
    backend:       Box<dyn StorageBackend>,
    root_path:     PathBuf,
}

impl Volume {
    // ── Create a new ENFS volume ────────────────────────────────────────────────

    pub async fn create(
        root: &Path,
        model_name: &str,
        author: &str,
        modalities: Vec<Modality>,
        quantizations: Vec<QuantizationKind>,
        tier_override: Option<StorageTier>,
    ) -> EnfsResult<Self> {
        std::fs::create_dir_all(root).map_err(EnfsError::Io)?;

        let backend = open_backend(root, tier_override).await?;
        let tier = backend.tier();

        // Generate volume UUID
        let uuid_bytes = *uuid::Uuid::new_v4().as_bytes();

        // Size: use 1/4 of reported throughput in MiB as default block count
        let volume_size_blocks = 256_000u64; // ~1 GiB at 4 KiB blocks

        let header = EnfsHeader::new(volume_size_blocks, tier, uuid_bytes);

        let mut manifest = EnfsManifest::new(model_name, author, modalities, quantizations);
        manifest.seal()?;

        // Write ENFS directory skeleton
        Self::write_directory_skeleton(root)?;

        let vol = Self {
            header,
            manifest,
            inodes:    InodeTable::new(),
            index:     TensorIndex::new(),
            allocator: BlockAllocator::new(volume_size_blocks),
            memory:    MemoryHierarchy::new(),
            backend,
            root_path: root.to_path_buf(),
        };

        vol.flush_header().await?;
        vol.flush_manifest().await?;

        Ok(vol)
    }

    // ── Open an existing ENFS volume ────────────────────────────────────────────

    pub async fn open(root: &Path) -> EnfsResult<Self> {
        let backend = open_backend(root, None).await?;

        // Read header from offset 0
        let header_bytes = backend.read_at(0, 256).await?;
        let header = EnfsHeader::from_bytes(&header_bytes)?;

        // Read manifest
        let manifest_bytes = backend.read_at(header.manifest_offset, 8192).await?;
        let manifest = EnfsManifest::from_bytes(&manifest_bytes)?;

        // Read tensor index
        let index_bytes = backend.read_at(header.index_offset, 256 * 1024).await?;
        let index = TensorIndex::from_bytes(&index_bytes)
            .unwrap_or_else(|_| TensorIndex::new());

        let allocator = BlockAllocator::new(header.volume_size_blocks);

        Ok(Self {
            header,
            manifest,
            inodes:    InodeTable::new(),
            index,
            allocator,
            memory:    MemoryHierarchy::new(),
            backend,
            root_path: root.to_path_buf(),
        })
    }

    // ── Tensor I/O ─────────────────────────────────────────────────────────────

    /// Write a tensor by name into a given domain.
    pub async fn write_tensor(
        &mut self,
        name: &str,
        domain: DomainTag,
        dtype: TensorDtype,
        shape: &[u64],
        data: &[u8],
    ) -> EnfsResult<u64> {
        // Allocate inode
        let mut inode = Inode::new_tensor(
            0, // placeholder — filled by allocate()
            name,
            dtype,
            shape,
            data.len() as u64,
        );

        // Compress block
        let codec = self.choose_codec(data.len());
        let compressed = compress(codec, data)?;

        // Allocate block
        let block_id = self.allocator.alloc()?;
        let block_offset = self.header.data_region_offset
            + block_id * self.header.block_size_bytes as u64;

        let block_hdr = BlockHeader::new(
            block_id,
            BlockKind::TensorData,
            codec,
            &compressed,
            data.len() as u32,
        );

        // Write: [u32 header_len][bincode(BlockHeader)][compressed payload]
        // The u32 prefix lets us read back the exact bincode size on any platform.
        let mut block_bytes: Vec<u8> = Vec::new();
        let hdr_bytes = bincode::serialize(&block_hdr)
            .map_err(|e| EnfsError::Serialization(e.to_string()))?;
        let hdr_len = hdr_bytes.len() as u32;
        block_bytes.extend_from_slice(&hdr_len.to_le_bytes());
        block_bytes.extend_from_slice(&hdr_bytes);
        block_bytes.extend_from_slice(&compressed);
        self.backend.write_at(block_offset, &block_bytes).await?;

        // Finalize inode
        inode.block_count = 1;
        inode.block_list_offset = block_offset;
        inode.finalize_hash(data);

        let inode_id = self.inodes.allocate(inode);
        self.index.insert(name, inode_id, domain);

        Ok(inode_id)
    }

    /// Read a tensor by name.
    pub async fn read_tensor(&self, name: &str) -> EnfsResult<Vec<u8>> {
        let inode_id = self.index.lookup(name)?;
        let inode = self.inodes.get(inode_id)?;

        // Read: first 4 bytes = u32 header_len, then header, then payload
        let block_offset = inode.block_list_offset;
        let meta = self.backend.read_at(block_offset, 4).await?;
        let hdr_len = u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]]) as usize;

        let total = 4 + hdr_len + inode.size_bytes as usize + 64; // +64 slack for compression overhead
        let block_bytes = self.backend.read_at(block_offset, total).await?;

        let hdr: BlockHeader = bincode::deserialize(&block_bytes[4..4 + hdr_len])
            .map_err(|e| EnfsError::Serialization(e.to_string()))?;

        let payload_start = 4 + hdr_len;
        let payload = &block_bytes[payload_start..payload_start + hdr.payload_len as usize];

        // Verify block integrity
        hdr.verify(payload)?;

        // Decompress
        let data = decompress(hdr.codec, payload)?;

        // Verify inode payload integrity
        inode.verify_payload(&data)?;

        Ok(data)
    }

    // ── Flush ─────────────────────────────────────────────────────────────────

    pub async fn flush(&self) -> EnfsResult<()> {
        self.backend.flush().await?;
        Ok(())
    }

    pub fn throughput_mbps(&self) -> u32 {
        self.backend.throughput_mbps()
    }

    pub fn storage_tier(&self) -> StorageTier {
        self.backend.tier()
    }

    pub fn block_size(&self) -> u32 {
        self.backend.block_size()
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    pub fn free_blocks(&self) -> u64 {
        self.allocator.free_count()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn choose_codec(&self, data_len: usize) -> BlockCodec {
        match self.backend.tier() {
            // NVMe/RAM: I/O is fast, skip compression overhead
            StorageTier::NVMe | StorageTier::Ram => BlockCodec::None,
            // SSD: LZ4 for minimal CPU overhead
            StorageTier::Ssd => {
                if data_len > 64 * 1024 { BlockCodec::Lz4 } else { BlockCodec::None }
            }
            // HDD: Zstd to reduce data transferred over slow link
            StorageTier::Hdd | StorageTier::Auto => {
                if data_len > 16 * 1024 { BlockCodec::Zstd } else { BlockCodec::Lz4 }
            }
        }
    }

    async fn flush_header(&self) -> EnfsResult<()> {
        let bytes = self.header.to_bytes()?;
        self.backend.write_at(0, &bytes).await
    }

    async fn flush_manifest(&self) -> EnfsResult<()> {
        let bytes = self.manifest.to_bytes()?;
        self.backend.write_at(self.header.manifest_offset, &bytes).await
    }

    /// Create the human-unreadable ENFS directory skeleton.
    fn write_directory_skeleton(root: &Path) -> EnfsResult<()> {
        // These directories hold binary data files only.
        let dirs = [
            "t",      // tensors
            "d/la",   // domain: language
            "d/ma",   // domain: mathematics
            "d/ph",   // domain: physics
            "d/sc",   // domain: science
            "d/sk",   // domain: skills
            "d/2d",   // domain: 2D perception
            "d/3d",   // domain: 3D perception
            "d/au",   // domain: audio
            "d/vi",   // domain: video
            "d/sf",   // domain: safety
            "d/in",   // domain: intelligence
            "a",      // adapters
            "c",      // cache
            "cp",     // checkpoints
            "sg",     // signatures
            "p",      // plugins
            "x",      // index
        ];
        for dir in dirs {
            std::fs::create_dir_all(root.join(dir)).map_err(EnfsError::Io)?;
        }

        // Write a binary-only marker file so OS file managers show no text
        let marker: [u8; 8] = [0x45, 0x4E, 0x46, 0x53, 0x00, 0x00, 0x00, 0x01];
        std::fs::write(root.join(".enfs"), marker).map_err(EnfsError::Io)?;

        Ok(())
    }
}
