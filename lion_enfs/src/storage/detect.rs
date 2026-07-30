// lion_enfs/src/storage/detect.rs — Storage Tier Auto-Detection
//
// Heuristically classifies the underlying block device as HDD / SSD / NVMe / RAM.
// Works on Linux via /sys/block. Falls back to SSD on other platforms.

use std::path::Path;
use crate::header::StorageTier;

/// Detect the storage tier of the filesystem path.
/// Returns `StorageTier::Ssd` when detection is inconclusive.
pub fn detect_storage_tier(path: &Path) -> StorageTier {
    #[cfg(target_os = "linux")]
    {
        if let Some(tier) = linux_detect(path) {
            return tier;
        }
    }
    // Default fallback
    StorageTier::Ssd
}

/// Benchmark observed sequential read speed and return matching tier.
/// This is a coarse latency probe — not a precise benchmark.
pub fn benchmark_throughput_mbps(path: &Path) -> u32 {
    let probe_size = 4 * 1024 * 1024usize; // 4 MiB probe
    let probe_path = path.join(".enfs_probe_tmp");

    // Write probe file
    let data = vec![0xABu8; probe_size];
    if std::fs::write(&probe_path, &data).is_err() {
        return StorageTier::Ssd.throughput_mbps();
    }

    let start = std::time::Instant::now();
    let read_result = std::fs::read(&probe_path);
    let elapsed = start.elapsed();

    let _ = std::fs::remove_file(&probe_path);

    if let Ok(read_data) = read_result {
        if read_data.len() == probe_size && elapsed.as_secs_f64() > 0.0 {
            let mb = probe_size as f64 / (1024.0 * 1024.0);
            let mbps = (mb / elapsed.as_secs_f64()) as u32;
            return mbps;
        }
    }

    StorageTier::Ssd.throughput_mbps()
}

/// Map observed throughput to a StorageTier.
pub fn tier_from_throughput(mbps: u32) -> StorageTier {
    match mbps {
        0..=299     => StorageTier::Hdd,
        300..=1_499 => StorageTier::Ssd,
        1_500..=9_999 => StorageTier::NVMe,
        _           => StorageTier::Ram,
    }
}

// ── Linux-specific detection via /sys/block ────────────────────────────────────

#[cfg(target_os = "linux")]
fn linux_detect(path: &Path) -> Option<StorageTier> {
    // Resolve the real path
    let canonical = std::fs::canonicalize(path).ok()?;
    let path_str = canonical.to_str()?;

    // Try to find the mount device via /proc/mounts
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    let mut best_device: Option<String> = None;
    let mut best_len = 0usize;

    for line in mounts.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 { continue; }
        let mount_point = parts[1];
        if path_str.starts_with(mount_point) && mount_point.len() > best_len {
            best_device = Some(parts[0].to_string());
            best_len = mount_point.len();
        }
    }

    let device = best_device?;

    // Extract device name (e.g. /dev/nvme0n1 → nvme0n1)
    let dev_name = device.trim_start_matches("/dev/");
    // Strip partition suffix (nvme0n1p1 → nvme0n1, sda1 → sda)
    let block_dev = strip_partition_suffix(dev_name);

    // Check rotational flag: 1 = HDD, 0 = SSD/NVMe
    let rotational_path = format!("/sys/block/{}/queue/rotational", block_dev);
    let rotational = std::fs::read_to_string(&rotational_path)
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(1);

    if rotational == 1 {
        return Some(StorageTier::Hdd);
    }

    // Distinguish NVMe from SATA SSD by device name prefix
    if block_dev.starts_with("nvme") {
        return Some(StorageTier::NVMe);
    }

    // Check if it's a RAM-backed device (tmpfs, ramfs, zram)
    let mounts2 = std::fs::read_to_string("/proc/mounts").ok()?;
    for line in mounts2.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let fs_type = parts[2];
            if matches!(fs_type, "tmpfs" | "ramfs") && path_str.starts_with(parts[1]) {
                return Some(StorageTier::Ram);
            }
        }
    }

    Some(StorageTier::Ssd)
}

#[cfg(target_os = "linux")]
fn strip_partition_suffix(dev: &str) -> &str {
    // nvme0n1p3 → nvme0n1
    // sda1 → sda
    let bytes = dev.as_bytes();
    let mut end = bytes.len();
    // Trim trailing digits (partition number)
    while end > 0 && bytes[end - 1].is_ascii_digit() {
        end -= 1;
    }
    // For NVMe devices strip the trailing 'p' that precedes the partition number
    if end > 0 && bytes[end - 1] == b'p' && dev.starts_with("nvme") {
        end -= 1;
    }
    &dev[..end]
}
