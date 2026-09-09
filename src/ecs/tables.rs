use crate::{
    RenderGameObject, RenderText,
    drw::drawable::DescriptorID,
    fnt::font::TextFont,
    game::GameObject,
    geom::{matrix::Transform, shapes::Shapes},
    mem::engine_memory::EngineMemory,
    res::assets::TextureStorage,
};
use color::Rgba8;
use hecs::CommandBuffer;
pub use hecs::{Bundle, ComponentError, Entity, World};
use rayon::ThreadPool;
use std::{
    any::{TypeId, type_name},
    io::Cursor,
    sync::Arc,
};
use tracing::debug;
use vulkano::image::sampler::Sampler;

pub type DynObject = Box<dyn DynamicallyObjectAlias>;

pub trait DynamicallyObjectAlias: GameObject + Send + Sync {}
impl<T> DynamicallyObjectAlias for T where T: GameObject + Send + Sync {}

pub struct EntityComponent {
    pub(crate) buffer: CommandBuffer,
    memory: Arc<EngineMemory>,
    thread_pool: Arc<ThreadPool>,
    sampler: Arc<Sampler>,
    fonts: Arc<TextFont>,
    storage: Arc<TextureStorage>,
}

impl EntityComponent {
    pub(crate) fn new(
        memory: Arc<EngineMemory>,
        sampler: Arc<Sampler>,
        thread_pool: Arc<ThreadPool>,
        fonts: Arc<TextFont>,
        storage: Arc<TextureStorage>,
    ) -> Self {
        Self {
            buffer: CommandBuffer::new(),
            memory,
            sampler,
            thread_pool,
            fonts,
            storage,
        }
    }

    pub fn add<G>(&mut self, drw: G, transformation: Transform, color: Rgba8)
    where
        G: GameObject + RenderGameObject + Send + Sync + 'static,
    {
        // TODO Here we can use thread pool for parallel push in world
        let class = ClassInfo::of::<G>();

        let shape = drw.shape();

        let boxed_drw: DynObject = Box::new(drw);

        self.push(boxed_drw, transformation, Some(shape), class, color);
        // DescriptorSet,
        // ClassInfo,
        // Shapes,
        // DynObject
    }

    pub fn add_with_class(
        &mut self,
        drw: DynObject,
        transformation: Transform,
        shape: Option<Shapes>,
        class: ClassInfo,
        colour: Rgba8,
    ) {
        self.push(drw, transformation, shape, class, colour);
    }

    pub fn add_text<G>(&mut self, drw: G, transformation: Transform, text_color: Rgba8)
    where
        G: GameObject + RenderText + Send + Sync + 'static,
    {
        let sprite_text_info = drw.info();
        let class_name: String = format!("{}::{}", type_name::<G>(), &sprite_text_info.text);
        let glphs = self.fonts.get_glyphs(sprite_text_info, text_color);

        if let Some(glyphs) = glphs {
            let image_buffer = glyphs.into_owned();

            let mut png = Vec::new();

            let mut cursor = Cursor::new(&mut png);

            image_buffer
                .write_to(&mut cursor, image::ImageFormat::Png)
                .expect("Failed to write png");

            let s = Shapes::Image(self.storage.load_texture_from_bytes(&png));

            let boxed_drw: DynObject = Box::new(drw);

            self.push(
                boxed_drw,
                transformation,
                Some(s),
                ClassInfo::of_class::<G>(class_name),
                text_color,
            );
        }
    }

    fn push(
        &mut self,
        drw: DynObject,
        transformation: Transform,
        shape: Option<Shapes>,
        class: ClassInfo,
        color: Rgba8,
    ) {
        debug!("{:?}", class);

        let memory_allocator = self.memory.memory_allocator.clone();
        let descriptor_set_allocator = self.memory.descriptor_allocator.clone();
        let descriptor_set_cache = self.memory.descriptors.clone();
        let pipeline_cache = self.memory.pipelines.clone();
        let sampler = Some(self.sampler.clone());

        let descriptor_id = DescriptorID::from(&class);

        if let Some(shp) = shape {
            debug!("Object contains shape");
            let shape_clone = shp.clone();
            // TODO Likely someday it's will crash
            self.thread_pool.spawn(move || {
                shape_clone.create_descriptor(
                    descriptor_id,
                    memory_allocator,
                    descriptor_set_allocator,
                    descriptor_set_cache,
                    pipeline_cache,
                    sampler,
                );
            });

            self.buffer.spawn((class, transformation, shp, drw, color));
        } else {
            debug!("Object not contains shape");
            self.buffer.spawn((class, transformation, drw, color));
        }
    }

    pub fn remove<T>(&mut self, entity: Entity)
    where
        T: Bundle + 'static,
    {
        self.buffer.remove::<T>(entity);
    }

    #[allow(dead_code)]
    /// Update [`Entity`] through buffer
    pub(crate) fn attach_render_descriptor<G>(&mut self, entity: Entity, drw: G)
    where
        G: GameObject + RenderGameObject + Send + Sync + 'static,
    {
        let class = ClassInfo::of::<G>();

        let shape = drw.shape();

        shape.create_descriptor(
            DescriptorID::from(&class),
            self.memory.memory_allocator.clone(),
            self.memory.descriptor_allocator.clone(),
            self.memory.descriptors.clone(),
            self.memory.pipelines.clone(),
            Some(self.sampler.clone()),
        );

        let boxed_drw: DynObject = Box::new(drw);

        self.buffer.insert(entity, (shape, boxed_drw));
    }

    /// Executes commands from buffer immediately.
    ///
    /// # Note
    /// May cause unexpected errors
    pub fn execute_commands(&mut self, world: &mut World) {
        self.buffer.run_on(world);
    }
}

#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub type_id: TypeId,
    pub class_name: String,
}

impl ClassInfo {
    /// Returns [`ClassInfo`] with structure name as class name.
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            class_name: type_name::<T>().to_owned(),
        }
    }

    /// Returns [`ClassInfo`] with a given class name.
    ///
    /// It may create unique objects with unique descriptors.
    /// # Example
    ///
    /// ```rust
    /// #[game_object]
    /// struct SecondDrawable {}
    ///
    /// impl GameObject for SecondDrawable {
    ///     fn update(&mut self, _world: &mut EntityComponent) {}
    ///
    ///     fn start(&mut self, _world: &mut EntityComponent) {}
    /// }
    ///
    /// #[game_object]
    /// struct CustomDrawable {
    /// }
    ///
    /// impl GameObject for CustomDrawable {
    ///     fn update(&mut self, _world: &mut EntityComponent) {
    ///     }
    ///
    ///     fn start(&mut self, world: &mut EntityComponent) {
    ///         // Create an object with unique data
    ///         let drw = SecondDrawable {
    ///             shape: Shapes::Square(ShapeCreateInfo::default().with_radius(f32::MAX)),
    ///         };
    ///         let transform = ...
    ///         
    ///         // Create a second object with unique data
    ///         let drw2 = SecondDrawable {
    ///             shape: Shapes::Square(ShapeCreateInfo::default().with_radius(0.3)),
    ///         };
    ///
    ///         let transform2 = ...
    ///
    ///         // Since the classes drw and drw2 are the same. Data between drw2 and drw are the
    ///         // same and last object, which will be added(with same classes), will create data for
    ///         // all objects of this class
    ///         // world.add(drw2, transform2);
    ///         // world.add(drw, transform);
    ///         // To avoid this we can use [`ClassInfo::of_class`] function
    ///
    ///         let shape = drw.shape();
    ///         let shape2 = drw2.shape();
    ///         let boxed_drw = Box::new(drw);
    ///         let boxed_drw2 = Box::new(drw2);
    ///         world.add_with_class(boxed_drw2, transform2, shape2, ClassInfo::of_class::<SecondDrawable>("drw2")); // <-- Here we create a unique class
    ///         world.add_with_class(boxed_drw, transform, shape, ClassInfo::of_class::<SecondDrawable>("drw")); // <-- Here we create a unique class
    ///     }
    /// }
    /// ```
    pub fn of_class<T: 'static>(class: String) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            class_name: class,
        }
    }
}
