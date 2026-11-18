use winit::window::{Window, WindowAttributes};

pub struct Configuration {
    pub window_attributes: WindowAttributes,
}

impl Default for Configuration {
    fn default() -> Self {
        Self { window_attributes: Window::default_attributes() }
    }
}
