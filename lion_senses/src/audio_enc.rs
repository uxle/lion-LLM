// lion_senses/src/audio_enc.rs — WAV Audio Feature Extraction

use std::path::Path;
use tracing::debug;

#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub features:       Vec<f32>,
    pub total_samples:  usize,
    pub sample_rate:    u32,
    pub duration_secs:  f32,
    pub rms_energy:     f32,
    pub zero_cross_rate: f32,
    pub is_loud:        bool,
}

pub struct AudioEncoder {
    pub feature_size:     usize,
    pub raw_sample_count: usize,
    pub band_count:       usize,
}

impl Default for AudioEncoder {
    fn default() -> Self {
        Self { feature_size: 64, raw_sample_count: 40, band_count: 16 }
    }
}

impl AudioEncoder {
    pub fn encode_file(&self, path: &Path) -> Result<AudioFeatures, String> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| format!("WAV load error: {}", e))?;
        let spec         = reader.spec();
        let sample_rate  = spec.sample_rate;
        let num_channels = spec.channels as usize;

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>()
                .filter_map(|s| s.ok()).collect(),
            hound::SampleFormat::Int => {
                let max_v = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max_v)
                    .collect()
            }
        };

        let mono: Vec<f32> = if num_channels > 1 {
            raw.chunks(num_channels)
                .map(|ch| ch.iter().sum::<f32>() / num_channels as f32)
                .collect()
        } else { raw };

        Ok(self.extract_features(&mono, sample_rate))
    }

    fn extract_features(&self, mono: &[f32], sample_rate: u32) -> AudioFeatures {
        let n             = mono.len();
        let duration_secs = n as f32 / sample_rate.max(1) as f32;

        let rms = if n == 0 { 0.0 } else {
            (mono.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt()
        };

        let zcr = if n < 2 { 0.0 } else {
            mono.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f32 / n as f32
        };

        // Downsampled raw samples.
        let raw_count = self.raw_sample_count.min(n.max(1));
        let raw_samples: Vec<f32> = if n == 0 {
            vec![0.0; raw_count]
        } else {
            (0..raw_count).map(|i| mono[(i * n / raw_count).min(n - 1)]).collect()
        };

        // Band energies.
        let band_size = (n / self.band_count).max(1);
        let bands: Vec<f32> = (0..self.band_count).map(|b| {
            let start = b * band_size;
            let end   = ((b + 1) * band_size).min(n);
            let slice = &mono[start..end];
            if slice.is_empty() { 0.0 } else {
                (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32)
                    .sqrt().clamp(0.0, 1.0) * 2.0 - 1.0
            }
        }).collect();

        let mut features = Vec::with_capacity(self.feature_size);
        features.extend_from_slice(&raw_samples);
        features.push(rms * 2.0 - 1.0);
        features.push(zcr * 2.0 - 1.0);
        features.extend(bands);
        features.resize(self.feature_size, 0.0);

        debug!("Audio: {}Hz, {:.2}s, rms={:.3}, zcr={:.3}", sample_rate, duration_secs, rms, zcr);

        AudioFeatures {
            features,
            total_samples: n,
            sample_rate,
            duration_secs,
            rms_energy: rms,
            zero_cross_rate: zcr,
            is_loud: rms > 0.5,
        }
    }

    pub fn encode_to_size(&self, path: &Path, input_size: usize) -> Result<Vec<f32>, String> {
        let mut f = self.encode_file(path)?.features;
        f.resize(input_size, 0.0);
        Ok(f)
    }
}
