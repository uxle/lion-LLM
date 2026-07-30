// lion_enfs/src/error.rs — ENFS Error Types

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnfsError {
    #[error("Invalid ENFS magic bytes — not an ENFS volume")]
    InvalidMagic,

    #[error("Unsupported ENFS version: {major}.{minor}")]
    UnsupportedVersion { major: u16, minor: u16 },

    #[error("Volume integrity check failed: {reason}")]
    IntegrityFailure { reason: String },

    #[error("Block {block_id} not found in volume")]
    BlockNotFound { block_id: u64 },

    #[error("Inode {inode_id} not found")]
    InodeNotFound { inode_id: u64 },

    #[error("Tensor '{name}' not found in index")]
    TensorNotFound { name: String },

    #[error("Domain '{domain}' does not exist")]
    DomainNotFound { domain: String },

    #[error("Volume is full — no free blocks available")]
    VolumeFull,

    #[error("Memory tier error: {0}")]
    MemoryTierError(String),

    #[error("Compression failed: {0}")]
    CompressionError(String),

    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Storage backend error: {0}")]
    StorageError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Permission denied — ENFS data is AI-only binary format")]
    HumanReadDenied,

    #[error("Timeout — storage backend did not respond within {ms}ms")]
    Timeout { ms: u64 },
}

pub type EnfsResult<T> = Result<T, EnfsError>;
