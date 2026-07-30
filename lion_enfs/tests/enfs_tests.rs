// lion_enfs/tests/enfs_tests.rs — ENFS Integration Tests

use lion_enfs::{
    Volume, StorageTier, DomainTag, TensorDtype,
    Modality, QuantizationKind,
    WorkingMemory, ArchiveMemory,
    DomainMemory, DomainKey,
    SensoryMemory,
    BlockAllocator,
    TensorIndex,
    EnfsHeader, ENFS_MAGIC,
};

// ── Header ────────────────────────────────────────────────────────────────────

#[test]
fn test_header_magic_and_roundtrip() {
    let uuid = [0u8; 16];
    let h = EnfsHeader::new(1024, StorageTier::NVMe, uuid);
    assert_eq!(h.magic, ENFS_MAGIC);
    assert!(h.validate().is_ok());
    let bytes = h.to_bytes().unwrap();
    let h2 = EnfsHeader::from_bytes(&bytes).unwrap();
    assert_eq!(h.magic, h2.magic);
    assert_eq!(h.storage_tier, h2.storage_tier);
}

#[test]
fn test_storage_tier_throughput() {
    assert!(StorageTier::NVMe.throughput_mbps() > StorageTier::Ssd.throughput_mbps());
    assert!(StorageTier::Ssd.throughput_mbps()  > StorageTier::Hdd.throughput_mbps());
    assert!(StorageTier::Ram.throughput_mbps()  >= StorageTier::NVMe.throughput_mbps());
}

#[test]
fn test_storage_tier_block_size_ordering() {
    // HDD uses the largest block size to amortise seeks
    assert!(StorageTier::Hdd.optimal_block_size() > StorageTier::Ssd.optimal_block_size());
    assert!(StorageTier::Ssd.optimal_block_size() > StorageTier::NVMe.optimal_block_size());
}

// ── Block Allocator ───────────────────────────────────────────────────────────

#[test]
fn test_block_allocator_alloc_free() {
    let mut alloc = BlockAllocator::new(1000);
    let b1 = alloc.alloc().unwrap();
    let b2 = alloc.alloc().unwrap();
    assert_ne!(b1, b2);

    let free_before = alloc.free_count();
    alloc.free(b1);
    assert_eq!(alloc.free_count(), free_before + 1);
}

#[test]
fn test_block_allocator_exhaustion() {
    let mut alloc = BlockAllocator::new(200); // 200 blocks, ~first 94 reserved
    let mut ids = vec![];
    while let Ok(id) = alloc.alloc() {
        ids.push(id);
        if ids.len() > 1000 { break; } // safety
    }
    // Next alloc should fail
    assert!(alloc.alloc().is_err());
}

// ── Tensor Index ──────────────────────────────────────────────────────────────

#[test]
fn test_tensor_index_insert_lookup() {
    let mut idx = TensorIndex::new();
    idx.insert("layer_0.q_proj", 42, DomainTag::Tensors);
    assert_eq!(idx.lookup("layer_0.q_proj").unwrap(), 42);
    assert!(idx.lookup("nonexistent").is_err());
}

#[test]
fn test_tensor_index_domain_listing() {
    let mut idx = TensorIndex::new();
    idx.insert("en_embed", 1, DomainTag::Language);
    idx.insert("math_ffn", 2, DomainTag::Mathematics);
    idx.insert("phys_attn", 3, DomainTag::Physics);

    let lang = idx.list_domain(DomainTag::Language);
    assert!(lang.contains(&1));
    assert!(!lang.contains(&2));
}

#[test]
fn test_tensor_index_binary_roundtrip() {
    let mut idx = TensorIndex::new();
    idx.insert("tensor_a", 10, DomainTag::Tensors);
    idx.insert("tensor_b", 11, DomainTag::Audio);

    let bytes = idx.to_bytes().unwrap();
    let idx2 = TensorIndex::from_bytes(&bytes).unwrap();
    assert_eq!(idx2.lookup("tensor_a").unwrap(), 10);
    assert_eq!(idx2.lookup("tensor_b").unwrap(), 11);
}

// ── Memory Tiers ─────────────────────────────────────────────────────────────

#[test]
fn test_sensory_memory_tick_eviction() {
    let mut mem = SensoryMemory::new();
    let id = mem.push(vec![1, 2, 3]);
    assert!(mem.fetch(id).is_some());
    mem.tick();
    mem.tick();
    mem.tick(); // after 3 ticks (>2), slot evicted
    assert!(mem.fetch(id).is_none());
}

#[test]
fn test_working_memory_lru_eviction() {
    let mut mem = WorkingMemory::new(3);
    mem.store("a", vec![1]);
    mem.store("b", vec![2]);
    mem.store("c", vec![3]);
    mem.store("d", vec![4]); // evicts "a" (LRU)
    assert!(mem.fetch("a").is_none());
    assert!(mem.fetch("d").is_some());
}

#[test]
fn test_domain_memory_store_fetch() {
    let mut mem = DomainMemory::new();
    mem.store(DomainKey::Mathematics, "pi_embedding", vec![3, 1, 4, 1, 5]).unwrap();
    let fetched = mem.fetch(DomainKey::Mathematics, "pi_embedding").unwrap();
    assert_eq!(fetched, &[3u8, 1, 4, 1, 5]);
    assert!(mem.fetch(DomainKey::Physics, "pi_embedding").is_none());
}

#[test]
fn test_archive_memory_compress_decompress() {
    let mut arch = ArchiveMemory::new();
    let data = vec![42u8; 4096]; // compressible
    arch.store("cold_tensor", data.clone());
    let recovered = arch.fetch("cold_tensor").unwrap();
    assert_eq!(recovered, data);
    assert!(arch.compressed_size_bytes() < data.len()); // compression works
}

// ── Volume (filesystem) ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_volume_create_and_write_read_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("test_vol");

    let mut vol = Volume::create(
        &root,
        "test_model",
        "test_author",
        vec![Modality::Text],
        vec![QuantizationKind::F16],
        Some(StorageTier::NVMe),
    ).await.unwrap();

    assert_eq!(vol.storage_tier(), StorageTier::NVMe);

    // Write a tensor
    let tensor_data: Vec<u8> = (0u8..128).collect();
    vol.write_tensor(
        "embed.weight",
        DomainTag::Language,
        TensorDtype::F16,
        &[4096, 32],
        &tensor_data,
    ).await.unwrap();

    assert_eq!(vol.tensor_count(), 1);

    // Read back and verify
    let recovered = vol.read_tensor("embed.weight").await.unwrap();
    assert_eq!(recovered, tensor_data);
}

#[tokio::test]
async fn test_volume_throughput_tier_matches() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("tier_test");

    let vol = Volume::create(
        &root, "t", "a",
        vec![Modality::Text],
        vec![QuantizationKind::F32],
        Some(StorageTier::Ssd),
    ).await.unwrap();

    assert_eq!(vol.storage_tier(), StorageTier::Ssd);
    assert_eq!(vol.throughput_mbps(), StorageTier::Ssd.throughput_mbps());
}

#[tokio::test]
async fn test_volume_binary_directory_skeleton() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("skeleton_test");

    Volume::create(&root, "m", "a",
        vec![Modality::Text], vec![QuantizationKind::F16],
        Some(StorageTier::NVMe),
    ).await.unwrap();

    // Verify binary marker file exists
    let marker = std::fs::read(root.join(".enfs")).unwrap();
    assert_eq!(&marker[..4], b"ENFS");

    // Verify domain directories created
    assert!(root.join("d/la").exists());
    assert!(root.join("d/ma").exists());
    assert!(root.join("d/ph").exists());
}
