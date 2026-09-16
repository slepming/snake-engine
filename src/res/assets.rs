use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Debug,
    path::Path,
    sync::{Arc, RwLock},
    thread,
};

use anyhow::Error;
use rust_embed::Embed;
use tracing::debug;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    image::{Image, ImageCreateInfo, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::{drw::texture::Texture, geom::dimension::Dimension, mem::engine_memory::EngineMemory};

#[derive(Embed)]
#[folder = "assets/"]
pub(crate) struct Asset; // TODO: Binary file large

/// Texture storage
///
/// Thread safety
pub struct TextureStorage {
    pub(crate) queue: Arc<Queue>,
    pub(crate) memory_allocs: Arc<EngineMemory>,
    pub(crate) texture_pool: RwLock<HashMap<String, Arc<TextureHandler>>>,
}

#[derive(Debug)]
pub struct TextureHandler {
    pub(crate) view: Arc<ImageView>,
}

impl TextureStorage {
    /// Returns Arc<[`TextureHandler`]> from bytes
    pub fn load_texture_from_bytes(&self, bytes: &[u8]) -> Arc<TextureHandler> {
        let texture: Texture = { Texture::from_slice(bytes).unwrap() };

        let texture_handler = self.create_texture_handler(texture.dimension, &texture.image);

        texture_handler
    }

    /// Returns Copy on Write byte slice
    pub fn load<'a>(file_name: &Path) -> Result<Cow<'a, [u8]>, Error> {
        let file = file_name.to_string_lossy();
        match Asset::get(&file) {
            Some(f) => Ok(f.data),
            None => Err(Error::msg("The file is not available in the storage")),
        }
    }

    /// Load texture to texture pool. You can save to the pool in advance.
    ///
    /// # Note
    /// If you use this function right before [`load_texture_handler`]
    ///  you're texture will be load from [`load_texture_handler`] function synchronously.
    /// It's happening because OS scheduler gives the ability to start job for another thread too long.
    ///
    /// [`load_texture_handler`]: TextureStorage::load_texture_handler
    pub fn load_and_save_texture(self: Arc<Self>, file_name: &'static Path, internal: bool) {
        let file = file_name.file_name().unwrap().to_string_lossy().to_string();
        let file_name = file_name.to_str().unwrap();
        thread::spawn(move || {
            let mut texture_pool = self.texture_pool.write().unwrap();

            if texture_pool.contains_key(&file) {
                debug!("Texture in texture pool exists");
                return;
            }

            let _span = tracy_client::span!("Engine::load_texture");

            let texture: Texture = {
                if internal {
                    Texture::from_internal_assets(file_name).unwrap()
                } else {
                    Texture::from_file(file_name).unwrap()
                }
            };

            debug!(
                loaded_image_size = texture.image.len(),
                loaded_image_size_mb = texture.image.len() / 1024 / 1024,
            );

            let texture_handler = self.create_texture_handler(texture.dimension, &texture.image);
            texture_pool.insert(file.clone(), texture_handler);
        });
    }

    /// Returns Arc<[`TextureHandler`]> from file system
    pub fn load_texture_handler(
        &self,
        file_name: &'static Path,
        internal: bool,
    ) -> Arc<TextureHandler> {
        let _span = tracy_client::span!("Engine::load_texture");
        let file = file_name.file_name().unwrap().to_string_lossy().to_string();

        if let Some(texture) = self.texture_pool.read().unwrap().get(&file) {
            return texture.clone();
        }

        let mut texture_pool = self.texture_pool.write().unwrap();

        if let Some(texture) = texture_pool.get(&file) {
            return texture.clone();
        }

        let texture: Texture = {
            if internal {
                Texture::from_internal_assets(file_name.to_str().unwrap()).unwrap()
            } else {
                Texture::from_file(file_name.to_str().unwrap()).unwrap()
            }
        };

        debug!(
            loaded_image_size = texture.image.len(),
            loaded_image_size_mb = texture.image.len() / 1024 / 1024
        );

        let texture_handler = self.create_texture_handler(texture.dimension, &texture.image);
        texture_pool.insert(file.clone(), texture_handler);

        texture_pool
            .get(&file)
            .expect(format!("texture pool not contain {}", file).as_str())
            .clone()
    }

    fn create_texture_handler(&self, dimension: Dimension, texture: &[u8]) -> Arc<TextureHandler> {
        let mut uploads = AutoCommandBufferBuilder::primary(
            self.memory_allocs.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let image_view: Arc<ImageView> = {
            let upload_buffer = Buffer::new_slice::<u8>(
                self.memory_allocs.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (dimension.dimension.x * dimension.dimension.y * 4 as f32) as DeviceSize,
            )
            .unwrap();

            upload_buffer.write().unwrap().copy_from_slice(texture);

            let image = Image::new(
                self.memory_allocs.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    extent: [
                        dimension.dimension.x as u32,
                        dimension.dimension.y as u32,
                        1,
                    ],
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();
            ImageView::new_default(image).unwrap()
        };

        let _ = uploads
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let texture_handler = TextureHandler { view: image_view };
        Arc::new(texture_handler)
    }
}
