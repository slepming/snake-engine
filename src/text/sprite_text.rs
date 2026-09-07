use crate::Vector;

/// Text options
#[derive(Clone)]
pub struct SpriteTextCreateInfo {
    /// Text which will be draw on the screen
    pub text: String,
    /// Character size
    pub scale: Vector,
    /// Font index(temp)
    pub font: usize,
    /// Positional coordinates
    pub position: Vector,
}

impl SpriteTextCreateInfo {
    pub fn with_text(mut self, text: &'static str) -> Self {
        self.text = String::from(text);
        self
    }

    pub fn with_position(mut self, position: Vector) -> Self {
        self.position = position;
        self
    }

    pub fn font_index(mut self, index: usize) -> Self {
        self.font = index;
        self
    }

    pub fn with_scale(mut self, s: Vector) -> Self {
        self.scale = s;
        self
    }
}

impl Default for SpriteTextCreateInfo {
    fn default() -> Self {
        Self {
            text: Default::default(),
            scale: Vector::new(15.0, 7.5),
            font: 0,
            position: Default::default(),
        }
    }
}
