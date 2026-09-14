use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use hecs::World;
use rayon::ThreadPool;
use snake_macros::static_game_object;
use vulkano::{device::Queue, image::sampler::Sampler};
use winit::dpi::PhysicalPosition;

use crate::{
    Shapes,
    ecs::tables::{ClassInfo, DynObject, EntityComponent},
    fnt::font::TextFont,
    geom::{matrix::Transform, shapes::ShapeCreateInfo},
    mem::engine_memory::EngineMemory,
    res::assets::TextureStorage,
};

use color::Rgba8;

pub type Ecs = EntityComponent;

const BASIC_FONT: &'static str = "Fonts/freedom.otf";

pub(crate) struct GameContext {
    #[allow(dead_code)]
    pub storage: Arc<TextureStorage>,
    pub frames: u64,
    #[allow(dead_code)]
    pub fonts: Arc<TextFont>,
    pub mouse_position: Option<PhysicalPosition<f64>>,
    pub world: Arc<RwLock<World>>,
    pub world_buffer: EntityComponent,
}

impl GameContext {
    pub fn new(
        memory: Arc<EngineMemory>,
        queue: Arc<Queue>,
        sampler: Arc<Sampler>,
        thread_pool: Arc<ThreadPool>,
    ) -> Self {
        let world = Arc::new(RwLock::new(World::new()));
        let fonts = Arc::new(TextFont::new(BASIC_FONT));

        let storage = Arc::new(TextureStorage {
            queue: queue.clone(),
            memory_allocs: memory.clone(),
            texture_pool: RwLock::new(HashMap::new()),
        });

        let world_buffer = EntityComponent::new(
            memory.clone(),
            sampler.clone(),
            thread_pool,
            fonts.clone(),
            storage.clone(),
        );

        Self {
            storage,
            frames: 0,
            fonts,
            mouse_position: None,
            world: world,
            world_buffer: world_buffer,
        }
    }
}

/// Contains general functions for the operations of the object
pub trait GameObject {
    /// Executes every frame
    fn update(&mut self, world: &mut EntityComponent);

    fn start(&mut self, world: &mut EntityComponent);
}

pub enum CanvasCommand {
    CreateObject {
        object: DynObject,
        transform: Transform,
        shape: Option<Shapes>,
        class: ClassInfo,
        colour: Rgba8,
    },
}

/// The main structure, handle first object structure from user.
#[static_game_object]
pub struct Canvas {
    pub buffer: Vec<CanvasCommand>,
}

impl Canvas {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            shape: Shapes::Square(ShapeCreateInfo::default()),
        }
    }
}

impl GameObject for Canvas {
    fn update(&mut self, _world: &mut EntityComponent) {}

    fn start(&mut self, world: &mut EntityComponent) {
        for command in self.buffer.drain(..) {
            match command {
                CanvasCommand::CreateObject {
                    object,
                    transform,
                    shape,
                    class,
                    colour,
                } => world.add_with_class(object, transform, shape, class, colour),
            }
        }
    }
}
