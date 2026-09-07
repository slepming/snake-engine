use std::{borrow::Cow, path::Path};

use ab_glyph::{Font, FontVec, Glyph, Point, PxScale, ScaleFont, point};
use color::Rgba8;
use image::{DynamicImage, ImageBuffer, Rgba};
use tracing::info;

use crate::{res::assets::TextureStorage, text::sprite_text::SpriteTextCreateInfo};

/// Stores the necessary parameters for fonts
pub struct TextFont {
    /// Fonts
    fonts: Vec<FontVec>,
}

impl TextFont {
    /// Returns [`TextFont`] structure
    pub fn new(family: &'static str) -> Self {
        let path_to_font = Path::new(family);

        let font_in_bytes = TextureStorage::load(&path_to_font).unwrap();
        let font = FontVec::try_from_vec(font_in_bytes.into_owned()).unwrap();

        let mut fonts: Vec<FontVec> = Vec::with_capacity(1);

        fonts.push(font);

        info!("Font: {} imported", family);

        Self { fonts }
    }

    /// Adds a font
    pub fn add_font(&mut self, family: &'static str) {
        let path_to_font = Path::new(family);

        let font_in_bytes = TextureStorage::load(&path_to_font).unwrap();
        let font = FontVec::try_from_vec(font_in_bytes.into_owned()).unwrap();

        self.fonts.push(font);
    }

    /// Returns text sprite through buffer
    pub fn get_glyphs<'a>(
        &self,
        text: SpriteTextCreateInfo,
        color: Rgba8,
    ) -> Option<Cow<'a, ImageBuffer<Rgba<u8>, Vec<u8>>>> {
        if text.text.len() < 1 {
            return None;
        }

        let scale = PxScale {
            x: text.scale.x,
            y: text.scale.y,
        };

        let scaled_font = self.fonts[text.font].as_scaled(scale);

        let mut glyphs = Vec::with_capacity(text.text.len());
        layout_paragraph(scaled_font, point(0.0, 0.0), 999.0, &text.text, &mut glyphs);
        // to work out the exact size needed for the drawn glyphs we need to outline
        // them and use their `px_bounds` which hold the coords of their render bounds.
        let outlined: Vec<_> = glyphs
            .into_iter()
            // Note: not all layout glyphs have outlines (e.g. " ")
            .filter_map(|g| scaled_font.outline_glyph(g))
            .collect();

        // combine px_bounds to get min bounding coords for the entire layout
        let Some(all_px_bounds) = outlined
            .iter()
            .map(|g| g.px_bounds())
            .reduce(|mut b, next| {
                b.min.x = b.min.x.min(next.min.x);
                b.max.x = b.max.x.max(next.max.x);
                b.min.y = b.min.y.min(next.min.y);
                b.max.y = b.max.y.max(next.max.y);
                b
            })
        else {
            panic!("No outlined glyphs?")
        };

        // create a new rgba image using the combined px bound width and height
        let mut image =
            DynamicImage::new_rgba8(all_px_bounds.width() as _, all_px_bounds.height() as _)
                .to_rgba8();

        // Loop through the glyphs in the text, positing each one on a line
        for glyph in outlined {
            let bounds = glyph.px_bounds();
            // calc top/left ords in "image space"
            // image-x=0 means the *left most pixel*, equivalent to
            // px_bounds.min.x which *may be non-zero* (and similarly with y)
            // so `- px_bounds.min` converts the left-most/top-most to 0
            let img_left = bounds.min.x as u32 - all_px_bounds.min.x as u32;
            let img_top = bounds.min.y as u32 - all_px_bounds.min.y as u32;
            // Draw the glyph into the image per-pixel by using the draw closure
            glyph.draw(|x, y, v| {
                // Offset the position by the glyph bounding box
                let px = image.get_pixel_mut(img_left + x, img_top + y);
                // Turn the coverage into an alpha value (blended with any previous)
                *px = Rgba([
                    color.r,
                    color.g,
                    color.b,
                    px.0[3].saturating_add((v * 255.0) as u8),
                ]);
            });
        }

        Some(Cow::Owned(image))
    }
}

fn layout_paragraph<F, SF>(
    font: SF,
    position: Point,
    max_width: f32,
    text: &str,
    target: &mut Vec<Glyph>,
) where
    F: Font,
    SF: ScaleFont<F>,
{
    let v_advance = font.height() + font.line_gap();
    let mut caret = position + point(0.0, font.ascent());
    let mut last_glyph: Option<Glyph> = None;
    for c in text.chars() {
        if c.is_control() {
            if c == '\n' {
                caret = point(position.x, caret.y + v_advance);
                last_glyph = None;
            }
            continue;
        }
        let mut glyph = font.scaled_glyph(c);
        if let Some(previous) = last_glyph.take() {
            caret.x += font.kern(previous.id, glyph.id);
        }
        glyph.position = caret;

        last_glyph = Some(glyph.clone());
        caret.x += font.h_advance(glyph.id);

        if !c.is_whitespace() && caret.x > position.x + max_width {
            caret = point(position.x, caret.y + v_advance);
            glyph.position = caret;
            last_glyph = None;
        }

        target.push(glyph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_glyph_buffer() {
        let sprite_text = SpriteTextCreateInfo::default().with_text("Hello, world");

        let font = TextFont::new("Fonts/freedom.otf");

        assert!(
            font.get_glyphs(
                sprite_text,
                Rgba8 {
                    r: 255,
                    g: 255,
                    b: 255,
                    a: 255
                }
            )
            .expect("Message is none")
            .len()
                > 0
        )
    }
}
