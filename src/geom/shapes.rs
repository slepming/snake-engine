use std::sync::Arc;
use strum::{AsRefStr, IntoStaticStr};
use tracing::{debug, warn};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::Sampler;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::pipeline::Pipeline;

use crate::SnakeVertex;
use crate::drw::drawable::DescriptorID;
use crate::drw::texture::Texture;
use crate::res::assets::TextureHandler;
use crate::res::cache::{CacheProvider, DescriptorSetCache, PipelineCache};

pub const SQUARE_VERTEX: [SnakeVertex; 4] = [
    SnakeVertex {
        position: [-1.0, -1.0],
    },
    SnakeVertex {
        position: [1.0, -1.0],
    },
    SnakeVertex {
        position: [1.0, 1.0],
    },
    SnakeVertex {
        position: [-1.0, 1.0],
    },
];

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub struct CircleData {
    pub radius: f32,
    pub thickness: f32,
}

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub struct SquareData {
    pub corner_radius: f32,
}

#[derive(AsRefStr, IntoStaticStr, Clone)]
pub enum Shapes {
    Square(ShapeCreateInfo),
    Circle,
    Image(Arc<TextureHandler>),
}

#[derive(Clone, Debug)]
pub struct ShapeCreateInfo {
    /// Shape texture. Overlaid on shape through shader
    pub texture: Option<Texture>,
    /// Shape radiusUnavailable current
    ///
    /// # Supports
    /// Square, Circle
    pub radius: f32,
    /// Outline thickness. Current unsupported
    pub thickness: f32,
}

impl ShapeCreateInfo {
    /// Sets the texture of the object(if supported)
    pub fn with_texture(mut self, texture: Texture) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Sets the radius of the object(if supported)
    /// Currently radius must be lower than 1.0
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = rapier2d::na::clamp(radius, 0.0, 1.0);
        self
    }

    /// Sets the thickness of the object(if supported)
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness;
        self
    }
}

impl Default for ShapeCreateInfo {
    fn default() -> Self {
        Self {
            texture: Default::default(),
            radius: 0.0,
            thickness: Default::default(),
        }
    }
}

impl Shapes {
    /// Creates descriptor set
    pub fn create_descriptor(
        &self,
        descriptor_id: DescriptorID,
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_cache: Arc<DescriptorSetCache>,
        pipeline_cache: Arc<PipelineCache>,
        sampler: Option<Arc<Sampler>>,
    ) {
        #[cfg(feature = "profiling")]
        let _span = tracy_client::span!("Shapes::create_descriptor");
        //debug!(shape = self.as_ref().to_lowercase());
        let pipeline = pipeline_cache
            .get(self.as_ref().to_lowercase().as_ref())
            .expect("Pipeline not found");

        match self {
            Shapes::Square(sci) => {
                let buffer = Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    SquareData {
                        corner_radius: sci.radius,
                    },
                )
                .unwrap();

                let layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
                if layout.bindings().is_empty() {
                    panic!("Pipeline 'square' has no bindings. Did you forget to compile shaders?");
                }

                let descriptor_set = DescriptorSet::new(
                    descriptor_allocator.clone(),
                    layout,
                    [WriteDescriptorSet::buffer(0, buffer)],
                    [],
                )
                .unwrap();

                let _ = descriptor_set_cache.insert((descriptor_id.clone().id, descriptor_set));
            }
            Shapes::Circle => {
                let buffer = Buffer::from_data(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    CircleData {
                        radius: 0.05,
                        thickness: 0.001,
                    },
                )
                .unwrap();

                let layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
                if layout.bindings().is_empty() {
                    warn!("Pipeline 'circle' has no bindings. Did you forget to compile shaders?");
                }

                if descriptor_set_cache.contains(&descriptor_id.id) {
                    debug!("Descriptor set exists");
                    return;
                }

                debug!("Descriptor set created!");
                let descriptor_set = DescriptorSet::new(
                    descriptor_allocator.clone(),
                    layout,
                    [WriteDescriptorSet::buffer(0, buffer)],
                    [],
                )
                .unwrap();

                let _ = descriptor_set_cache.insert((descriptor_id.clone().id, descriptor_set));
            }
            Shapes::Image(texture) => {
                let layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
                if layout.bindings().is_empty() {
                    warn!("Pipeline 'image' has no bindings. Did you forget to compile shaders?");
                }
                let image_view = texture.view.clone();

                if descriptor_set_cache.contains(&descriptor_id.id) {
                    debug!("Descriptor set exists");
                    return;
                }

                let descriptor_set = DescriptorSet::new(
                    descriptor_allocator.clone(),
                    layout,
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        image_view,
                        sampler.unwrap().clone(),
                    )],
                    [],
                )
                .unwrap();

                debug!("Descriptor set created!");

                let _ = descriptor_set_cache.insert((descriptor_id.clone().id, descriptor_set));
            }
        }
    }
}
