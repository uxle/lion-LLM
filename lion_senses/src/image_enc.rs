// lion_senses/src/image_enc.rs — Image Feature Extraction
//
// Loads any image (PNG, JPG, BMP, GIF, TIFF), resizes to 8×8 grayscale,
// and produces a 64-element float feature vector for the TernaryEncoder.

use std::path::Path;
use image::{DynamicImage, GenericImageView};
use tracing::debug;

// =============================================================================
// IMAGE FEATURES
// =============================================================================

#[derive(Debug, Clone)]
pub struct ImageFeatures {
    /// Raw f32 features in [-1.0, +1.0]. Length = input_size (default 64).
    pub features:       Vec<f32>,
    pub original_width: u32,
    pub original_height: u32,
    pub color_mode:     String,
    /// Sobel edge energy [0.0, 1.0]. High = high contrast / lots of edges.
    pub edge_energy:    f32,
    pub is_high_contrast: bool,
}

// =============================================================================
// IMAGE ENCODER
// =============================================================================

pub struct ImageEncoder {
    pub target_w: u32,
    pub target_h: u32,
}

impl Default for ImageEncoder {
    fn default() -> Self {
        Self { target_w: 8, target_h: 8 }
    }
}

impl ImageEncoder {
    pub fn new(target_w: u32, target_h: u32) -> Self {
        Self { target_w, target_h }
    }

    /// Feature count: width × height + 3 (RGB means) + 1 (edge energy).
    pub fn feature_count(&self) -> usize {
        (self.target_w * self.target_h) as usize + 4
    }

    /// Loads and encodes a file path.
    pub fn encode_file(&self, path: &Path) -> Result<ImageFeatures, String> {
        let img = image::open(path).map_err(|e| format!("Image load error: {}", e))?;
        Ok(self.encode_image(&img))
    }

    /// Encodes raw bytes.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<ImageFeatures, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("Image decode error: {}", e))?;
        Ok(self.encode_image(&img))
    }

    pub fn encode_image(&self, img: &DynamicImage) -> ImageFeatures {
        let (orig_w, orig_h) = img.dimensions();
        let color_mode = if img.color().has_color() { "RGB" } else { "L" }.to_string();

        let resized = img.resize_exact(
            self.target_w, self.target_h,
            image::imageops::FilterType::Lanczos3,
        );

        // Grayscale pixel values → [-1, +1].
        let gray = resized.to_luma8();
        let mut features: Vec<f32> = gray.pixels()
            .map(|p| (p[0] as f32 / 127.5) - 1.0)
            .collect();

        // RGB channel means → [-1, +1] each.
        let rgb = resized.to_rgb8();
        let n   = rgb.pixels().count() as f32;
        let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
        for p in rgb.pixels() {
            r += p[0] as f32 / 255.0;
            g += p[1] as f32 / 255.0;
            b += p[2] as f32 / 255.0;
        }
        features.push((r / n) * 2.0 - 1.0);
        features.push((g / n) * 2.0 - 1.0);
        features.push((b / n) * 2.0 - 1.0);

        // Edge energy.
        let edge_energy = compute_edge_energy(&gray);
        features.push(edge_energy * 2.0 - 1.0);

        let is_high_contrast = edge_energy > 0.3;

        debug!(
            "Image encoded: {}×{} {} → {} features, edge={:.3}",
            orig_w, orig_h, color_mode, features.len(), edge_energy
        );

        ImageFeatures { features, original_width: orig_w, original_height: orig_h, color_mode, edge_energy, is_high_contrast }
    }

    /// Encodes to exactly `input_size` float features (padded or truncated).
    pub fn encode_to_size(&self, path: &Path, input_size: usize) -> Result<Vec<f32>, String> {
        let mut f = self.encode_file(path)?.features;
        f.resize(input_size, 0.0);
        Ok(f)
    }

    /// Encode to quantized i8 for the TernaryEncoder.
    pub fn encode_to_i8(&self, path: &Path, input_size: usize) -> Result<Vec<i8>, String> {
        let f = self.encode_to_size(path, input_size)?;
        Ok(f.iter().map(|&x| lion_core::f32_to_i8(x)).collect())
    }

    /// Returns base64-encoded image data for the Ollama vision API.
    pub fn to_base64(&self, path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        use base64::Engine;
        Ok(base64::engine::general_purpose::STANDARD.encode(&bytes))
    }
}

// =============================================================================
// SOBEL EDGE ENERGY
// =============================================================================

fn compute_edge_energy(gray: &image::GrayImage) -> f32 {
    let (w, h) = gray.dimensions();
    if w < 3 || h < 3 { return 0.0; }

    let mut total = 0.0f32;
    let count     = ((w - 2) * (h - 2)) as f32;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let px = |dx: i32, dy: i32| -> f32 {
                gray.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] as f32 / 127.5 - 1.0
            };

            let gx = -px(-1,-1) + px(1,-1) - 2.0*px(-1,0) + 2.0*px(1,0) - px(-1,1) + px(1,1);
            let gy = -px(-1,-1) - 2.0*px(0,-1) - px(1,-1) + px(-1,1) + 2.0*px(0,1) + px(1,1);
            total += (gx*gx + gy*gy).sqrt() / (8.0 * 2.0_f32.sqrt());
        }
    }

    (total / count).clamp(0.0, 1.0)
}
