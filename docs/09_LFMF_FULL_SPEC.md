# LFMF v1.0 — Lion Flexible Model Format Specification

**Version:** 1.0.0
**Status:** Design Specification
**Extension:** `.lfmf`
**MIME Type:** `application/x-lfmf`
**Magic Number:** `LFMF` — `0x4C 0x46 0x4D 0x46`

> **Design status note:** This is an architectural blueprint. The `LfmfHeader` struct,
> magic bytes, and basic tiered memory manager are implemented in `lion_core/src/lfmf.rs`.
> Features marked [PLANNED] are specified here but not yet in code.

---

## Design Goals

- One container format for every AI model type
- Independent, per-chunk compression and encryption
- Native multimodal support: text, image, audio, video, 3D
- Streaming and memory-mapped loading
- Delta updates and adapter (LoRA/QLoRA) packs
- Cryptographic integrity and optional encryption
- Cross-platform and runtime-agnostic

---

## File Layout

```
+--------------------+
| Header             |  Magic, version, alignment, flags, architecture
+--------------------+
| Manifest           |  Model identity, author, license, modalities
+--------------------+
| Metadata           |  Key-value pairs (context_length, hidden_size, etc.)
+--------------------+
| Tokenizer          |  SentencePiece / BPE / WordPiece / Custom
+--------------------+
| Vocabulary         |  Tokens, frequencies, merge rules, special tokens
+--------------------+
| Runtime Graph      |  Computation graph (Transformer, MoE, CNN, SSM, Hybrid)
+--------------------+
| Weight Index       |  Offset map for fast random shard access
+--------------------+
| Weight Chunks      |  Per-layer tensor shards (arbitrary order)
+--------------------+
| Adapter Packs      |  LoRA / QLoRA / DoRA / IA³ packs [PLANNED]
+--------------------+
| Assets             |  Non-tensor support data (images, audio clips, etc.)
+--------------------+
| Checksums          |  Per-chunk and whole-file hashes
+--------------------+
| Signature          |  Ed25519 or RSA-4096 digital signature
+--------------------+
```

---

## Header Fields

| Field | Type | Description |
|---|---|---|
| `magic` | `[u8; 4]` | `0x4C 0x46 0x4D 0x46` ("LFMF") |
| `version_major` | `u16` | Breaking change increment |
| `version_minor` | `u16` | Backward-compatible increment |
| `version_patch` | `u16` | Bug-fix increment |
| `alignment_bytes` | `u32` | Chunk alignment (64–16384 bytes) |
| `compression` | `u8` | Default: None, Zstd, LZ4, Brotli, Gzip, Snappy |
| `encryption` | `u8` | None / AES-256-GCM / ChaCha20-Poly1305 |
| `hash_algo` | `u8` | SHA-256 / SHA-512 / BLAKE3 (default: BLAKE3) |
| `endianness` | `u8` | 0 = Little, 1 = Big |
| `chunk_count` | `u64` | Total number of chunks |
| `flags` | `u64` | Feature bitmask |
| `model_name` | `str` | Null-terminated UTF-8 model name |

---

## Manifest Fields

```json
{
  "model_name": "string",
  "author": "string",
  "organization": "string",
  "license": "string",
  "created_date": "ISO 8601",
  "framework": "string",
  "training_version": "string",
  "runtime_version": "string",
  "language_count": 0,
  "modalities": ["text", "image", "audio", "video", "3d"],
  "quantizations": ["FP16", "INT8", "INT4", "NF4"],
  "required_runtime": "string"
}
```

---

## Metadata Key-Value Pairs (common examples)

| Key | Description |
|---|---|
| `context_length` | Maximum context window in tokens |
| `embedding_size` | Token embedding dimension |
| `hidden_size` | Hidden layer dimension |
| `layer_count` | Number of transformer layers |
| `head_count` | Attention head count |
| `kv_heads` | Key-value head count (GQA/MQA) |
| `vocab_size` | Vocabulary size |
| `parameter_count` | Total parameter count |
| `training_tokens` | Tokens seen during training |
| `recommended_ram` | Minimum RAM for inference |
| `recommended_vram` | Minimum VRAM for GPU inference |

---

## Supported Tensor Types

### Floating Point
`FP64` · `FP32` · `TF32` · `BF16` · `FP16` · `FP8`

### Integer (signed)
`INT64` · `INT32` · `INT16` · `INT8` · `INT4` · `INT2` · `INT1`

### Unsigned Integer
`UINT64` · `UINT32` · `UINT16` · `UINT8` · `UINT4` · `UINT2` · `UINT1`

### Quantized / Experimental
`NF4` · `Q2` · `Q3` · `Q4` · `Q5` · `Q6` · `Q8` · `MXFP4` · `MXFP8` · Custom

Each tensor stores: name, dtype, shape, stride, compression, encryption, byte offset, byte length, BLAKE3 checksum, flags.

**Mixed precision:** Different layers may use different dtypes within the same file — e.g., attention in FP16, MLP in INT4.

---

## Chunk Types

| Type | Purpose |
|---|---|
| `HEADER` | File header |
| `MANIFEST` | Model identity |
| `METADATA` | Key-value metadata |
| `TOKENIZER` | Tokenizer binary |
| `VOCAB` | Vocabulary data |
| `GRAPH` | Runtime computation graph |
| `INDEX` | Tensor offset index |
| `TENSOR` | Weight tensor shard |
| `IMAGE` | Embedded image asset |
| `AUDIO` | Embedded audio asset |
| `VIDEO` | Embedded video asset |
| `POINT_CLOUD` | 3D point cloud |
| `MESH` | 3D mesh (vertices, faces, normals) |
| `TEXT` | Embedded text asset |
| `LORA` | LoRA/QLoRA adapter pack |
| `PLUGIN` | Runtime extension plugin |
| `CHECKSUM` | Integrity checksums |
| `SIGNATURE` | Digital signature |
| `CUSTOM` | Vendor/user-defined chunk |

Unknown chunk types are **skipped safely** — older runtimes continue loading supported chunks.

---

## Multimodal Support

### 2D
Images (PNG, JPEG, WebP, raw), video frames, waveform data, spectrograms, depth maps.

### 3D
```
Vertices · Faces · Normals · Tangents · UV coordinates
Materials · Animations · Skeleton · Bone weights
```

### Sensor / Medical
Time series, sensor streams, medical imaging (DICOM-compatible metadata).

---

## Compression

Each chunk chooses independently:

| Algorithm | Trade-off |
|---|---|
| None | Zero overhead, max speed |
| LZ4 | Fastest compress/decompress |
| Zstd | Best ratio/speed balance |
| Brotli | Best ratio (slow compress) |
| Gzip | Widest compatibility |
| Snappy | Google-optimized fast path |
| Custom | Plugin-defined |

---

## Encryption [PLANNED]

| Algorithm | Use case |
|---|---|
| AES-256-GCM | Hardware-accelerated, standard |
| ChaCha20-Poly1305 | Software-fast, mobile-friendly |

Encryption is per-chunk, not whole-file, enabling selective tensor protection.

---

## Streaming & Loading Modes

| Mode | Description |
|---|---|
| Lazy Loading | Load chunks on demand |
| Layer Loading | Prefetch next N layers |
| Demand Loading | Load only requested tensors |
| Prediction Loading | Speculative prefetch based on execution graph |
| Remote Streaming | Fetch shards from object store |
| SSD Streaming | Direct NVMe streaming bypass |

---

## Adapter Support [PLANNED]

Supported adapter types within a single `.lfmf` file:
`LoRA` · `QLoRA` · `DoRA` · `IA³` · `Prompt Tuning` · `Prefix Tuning` · `Custom`

---

## Runtime Compatibility

CPU · CUDA · ROCm · Metal · OpenCL · Vulkan · DirectML · WebGPU · TPU · NPU · FPGA

---

## Security

| Feature | Implementation |
|---|---|
| Per-chunk hash | BLAKE3 (default) / SHA-256 / SHA-512 |
| Whole-file hash | BLAKE3 over all chunk hashes |
| Digital signature | Ed25519 (preferred) / RSA-4096 |
| Encrypted tensors | AES-256-GCM or ChaCha20-Poly1305 |
| Runtime verification | Hash checked on load before use |

---

## Delta Updates [PLANNED]

Stores only changed tensors relative to a base version. Useful for:
- Model fine-tuning deployment
- Enterprise weight updates
- LoRA merging without full repack

---

## Versioning Rules

- **Major** bump: breaking format change — old loaders reject the file.
- **Minor** bump: new chunk types added — old loaders skip unknown chunks safely.
- **Patch** bump: bug fixes, metadata corrections.

---

## LFMF vs. Existing Formats

| Feature | GGUF | Safetensors | LFMF |
|---|---|---|---|
| Single-file portability | ✅ | ✅ | ✅ |
| Multi-file sharding | ❌ | Partial | ✅ |
| Mixed precision per layer | ✅ | ❌ | ✅ |
| Native LoRA/adapter packs | ❌ | ❌ | ✅ |
| Multimodal (image/audio/3D) | ❌ | ❌ | ✅ |
| Delta updates | ❌ | ❌ | ✅ |
| Per-chunk encryption | ❌ | ❌ | ✅ |
| Digital signature | ❌ | ❌ | ✅ |
| mmap zero-copy loading | ✅ | ✅ | ✅ |
| Stream loading | ❌ | ❌ | ✅ |
| Unknown chunk skipping | ❌ | ❌ | ✅ |

---

## Loading Flow

```
User
 └─→ Loader: open file
      └─→ Read header + manifest
      └─→ Verify BLAKE3 checksums + signature
      └─→ Memory-map required shards only
      └─→ Load tokenizer + metadata
      └─→ Build runtime graph
      └─→ Engine: ready for inference
```

---

## Roadmap

| Version | Focus |
|---|---|
| v1.0 | Core container, tensors, metadata, tokenizer, multimodal support |
| v2.0 | Distributed inference, remote tensor streaming, advanced quantization |
| v3.0 | Native MoE routing, federated model packages, secure execution metadata, cross-model composition |

---

## Relationship to ENFS

LFMF is the **packed distribution format** (a single `.lfmf` file).
ENFS is the **development filesystem** (a directory tree the loader reads from).

A toolchain converts between them:

```
model.enfs/  ──pack──→  model.lfmf   (for distribution)
model.lfmf   ──unpack──→  model.enfs/ (for editing/inspection)
```
