#![deny(warnings)]

pub use color::Rgba8;
pub use hecs::{CommandBuffer, Entity};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    ops::RangeInclusive,
    sync::{Arc, RwLock},
};
use tracing::debug;

use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    device::{Device, DeviceExtensions, Queue},
    image::{
        ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    },
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        Pipeline,
        graphics::{
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            vertex_input::Vertex,
            viewport::Viewport,
        },
    },
    swapchain::{
        CompositeAlpha, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
        acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalSize, Size},
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    platform::modifier_supplement::KeyEventExtModifierSupplement,
    window::{Fullscreen, Window, WindowId},
};

#[cfg(target_family = "unix")]
use winit::platform::wayland::WindowAttributesExtWayland;

use crate::{
    dbg::debug_utils::DebugUtils,
    ecs::tables::{ClassInfo, DynObject},
    game::{Canvas, GameContext, GameObject},
    geom::{
        matrix::Transform,
        shapes::{SQUARE_VERTEX, Shapes},
    },
    mem::engine_memory::EngineMemory,
    render::{MeshBuffers, RenderContext},
    res::cache::{CacheProvider, DescriptorSetCache, PipelineCache},
    shaders::{
        circle_shader::{circle_fs, circle_vs},
        image_shader::{image_fs, image_vs},
        square_shader::{square_fs, square_vs},
    },
    text::sprite_text::SpriteTextCreateInfo,
    threading::scheduler::{Scheduler, SchedulerContext, create_scheduler},
    utils::{
        vulkan::{create_pipeline, get_vulkan_instance, select_render_device},
        window::window_size_dependent_setup,
    },
};

pub use snake_macros::{static_game_object, text_object};

//#[cfg(debug_assertions)]
//use crate::testing::finder::Finder;

pub mod dbg;
pub mod drw;
pub mod ecs;
pub mod fnt;
pub mod game;
pub mod geom;
pub mod mem;
pub mod mv;
pub mod render;
pub mod res;
pub mod shaders;
pub mod testing;
pub mod text;
pub mod threading;
pub mod utils;

pub type Vector = glam::Vec2;
pub type GameObjectDrawable = Arc<RwLock<Box<dyn GameObject>>>;

const THREAD_POOL_SIZE: usize = 3;

#[global_allocator]
#[cfg(debug_assertions)]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 5);

/// The main entry point into the engine
pub struct EngineContext {
    instance: Arc<Instance>,
    /// One of the most important parts of the engine
    device: Arc<Device>,
    queues: Vec<Arc<Queue>>,
    #[allow(dead_code)]
    sampler: Arc<Sampler>,
    rcx: Option<RenderContext>,
    memory: Arc<EngineMemory>,
    game: GameContext,
    #[allow(dead_code)]
    debug: DebugUtils,
    #[allow(dead_code)]
    thread_pool: Arc<ThreadPool>,
    pipelines: Arc<PipelineCache>,
    descriptors: Arc<DescriptorSetCache>,
    pub scheduler: (Scheduler, Arc<SchedulerContext>),
}

impl EngineContext {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        tracing_subscriber::fmt::init();

        let _span = tracy_client::span!("Engine::new");

        let instance = get_vulkan_instance(event_loop, vec![]);
        let debug: DebugUtils = DebugUtils::new(instance.clone());

        // Choose device extensions that we're going to use. In order to present images to a
        // surface, we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (device, queues) =
            select_render_device(instance.clone(), device_extensions, event_loop);

        let queues = queues.collect::<Vec<_>>();

        let memory = Arc::new(EngineMemory::new(device.clone()));
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: vulkano::image::sampler::BorderColor::FloatTransparentBlack,
                ..Default::default()
            },
        )
        .unwrap();

        let descriptorset_cache = Arc::new(DescriptorSetCache::default());
        let pipeline_cache = Arc::new(PipelineCache::default());

        let thread_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(THREAD_POOL_SIZE)
                .build()
                .expect("Cannot create thread pool"),
        );

        let game = GameContext::new(
            memory.clone(),
            queues.last().expect("queues size 0").clone(),
            pipeline_cache.clone(),
            descriptorset_cache.clone(),
            sampler.clone(),
            thread_pool.clone(),
        );

        let canvas = Canvas::new();
        let class = ClassInfo::of::<Canvas>();

        game.world.write().unwrap().spawn((class, canvas));

        Self {
            game: game,
            descriptors: descriptorset_cache.clone(),
            pipelines: pipeline_cache.clone(),
            memory,
            instance,
            device,
            queues,
            sampler,
            rcx: None,
            debug,
            thread_pool,
            scheduler: create_scheduler(),
        }
    }

    /// Adds first object to engine. It's object always is space. Space has no colour, shape
    /// and position
    pub fn add_object<T>(&mut self, object: T, transform: Transform)
    where
        T: GameObject + Send + Sync + 'static,
    {
        let mut world = self.game.world.write().unwrap();
        let canvas_f = world.query_mut::<&mut Canvas>();

        let mut canvas_v: Vec<&mut Canvas> = canvas_f.into_iter().collect();

        let canvas = canvas_v.iter_mut().next();

        let class = ClassInfo::of::<T>();

        let object_boxed = Box::new(object);

        canvas
            .unwrap()
            .buffer
            .push(game::CanvasCommand::CreateObject {
                object: object_boxed,
                transform,
                shape: None,
                class,
                colour: Rgba8 {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 0,
                },
            });
    }
}

impl ApplicationHandler for EngineContext {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let _span = tracy_client::span!("Engine::resumed");
        let window: Arc<Window>;
        debug!("Creating window");
        #[cfg(target_family = "unix")]
        {
            window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("snake")
                            .with_name("snake-engine", "snake-engine")
                            .with_fullscreen(Some(Fullscreen::Borderless(None)))
                            .with_min_inner_size(Size::Physical(PhysicalSize {
                                width: 640,
                                height: 480,
                            }))
                            .with_max_inner_size(
                                event_loop.available_monitors().next().unwrap().size(),
                            ),
                    )
                    .unwrap(),
            );
        }

        #[cfg(target_family = "windows")]
        {
            window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("snake")
                            .with_fullscreen(Some(Fullscreen::Borderless(None)))
                            .with_min_inner_size(Size::Physical(PhysicalSize {
                                width: 640,
                                height: 480,
                            }))
                            .with_max_inner_size(
                                event_loop.available_monitors().next().unwrap().size(),
                            ),
                    )
                    .unwrap(),
            );
        }

        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        // Before we can draw on the surface, we have to create what is called a swapchain.
        // Creating a swapchain allocates the color buffers that will contain the image that will
        // ultimately be visible on the screen. These images are returned alongside the swapchain.
        let (swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            // Composite alpha priority
            let priority_hierarchy = [
                CompositeAlpha::PreMultiplied,
                CompositeAlpha::PostMultiplied,
                CompositeAlpha::Inherit,
                CompositeAlpha::Opaque,
            ];

            let supported_composite_alpha = priority_hierarchy
                .into_iter()
                .find(|&desired_alpha| {
                    surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .any(|available_alpha| available_alpha == desired_alpha)
                })
                .unwrap_or_else(|| {
                    surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .expect("Device don't support CompositeAlpha")
                });

            let supported_present_modes = self
                .device
                .physical_device()
                .surface_present_modes(&surface, SurfaceInfo::default())
                .unwrap();

            debug!("Supported present modes: {:?}", supported_present_modes);

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
                self.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    // Some drivers report an `min_image_count` of 1, but fullscreen mode requires
                    // at least 2. Therefore we must ensure the count is at least 2, otherwise the
                    // program would crash when entering fullscreen mode on those drivers.
                    min_image_count: surface_capabilities.min_image_count.max(2),

                    image_format,

                    // The size of the window, only used to initially setup the swapchain.
                    //
                    // NOTE:
                    // On some drivers the swapchain extent is specified by
                    // `surface_capabilities.current_extent` and the swapchain size must use this
                    // extent. This extent is always the same as the window size.
                    //
                    // However, other drivers don't specify a value, i.e.
                    // `surface_capabilities.current_extent` is `None`. These drivers will allow
                    // anything, but the only sensible value is the window size.
                    //
                    // Both of these cases need the swapchain to use the window size, so we just
                    // use that.
                    image_extent: window_size.into(),

                    image_usage: ImageUsage::COLOR_ATTACHMENT,

                    // The alpha mode indicates how the alpha value of the final image will behave.
                    // For example, you can choose whether the window will be
                    // opaque or transparent.
                    composite_alpha: supported_composite_alpha,
                    present_mode: supported_present_modes
                        .first()
                        .unwrap_or(&vulkano::swapchain::PresentMode::Mailbox)
                        .clone(),

                    ..Default::default()
                },
            )
            .unwrap()
        };

        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images where the
        // colors, depth and/or stencil information will be written.
        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `format: <ty>` indicates the type of the format of the image. This has to be
                    // one of the types of the `vulkano::format` module (or alternatively one of
                    // your structs that implements the `FormatDesc` trait). Here we use the same
                    // format as the swapchain.
                    format: swapchain.image_format(),
                    // `samples: 1` means that we ask the GPU to use one sample to determine the
                    // value of each pixel in the color attachment. We could use a larger value
                    // (multisampling) for antialiasing. An example of this can be found in
                    samples: 1,
                    // `load_op: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load_op: Clear,
                    // `store_op: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store_op: Store,
                },
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {},
            },
        )
        .unwrap();

        // The render pass we created above only describes the layout of our framebuffers. Before
        // we can draw we also need to create the actual framebuffers.
        //
        // Since we need to draw to multiple images, we are going to create a different framebuffer
        // for each image.
        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: RangeInclusive::new(0.0, 1.0),
        };

        // In some situations, the swapchain will become invalid by itself. This includes for
        // example when the window is resized (as the images of the swapchain will no longer match
        // the window's) or, on Android, when the application went to the background and goes back
        // to the foreground.
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not
        // work. To continue rendering, we need to recreate the swapchain by creating a new
        // swapchain. Here, we remember that we need to do this for the next loop iteration.
        let recreate_swapchain = false;

        // In the `window_event` handler below we are going to submit commands to the GPU.
        // Submitting a command produces an object that implements the `GpuFuture` trait, which
        // holds the resources for as long as they are in use by the GPU.
        //
        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to
        // avoid that, we store the submission of the previous frame here.
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let vs_square = square_vs::load(self.device.clone()).unwrap();
        let fs_square = square_fs::load(self.device.clone()).unwrap();
        let square_pipeline = create_pipeline(
            self.device.clone(),
            render_pass.clone(),
            vs_square.entry_point("main").unwrap(),
            fs_square.entry_point("main").unwrap(),
            ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        let vs_circle = circle_vs::load(self.device.clone()).unwrap();
        let fs_circle = circle_fs::load(self.device.clone()).unwrap();
        let circle_pipeline = create_pipeline(
            self.device.clone(),
            render_pass.clone(),
            vs_circle.entry_point("main").unwrap(),
            fs_circle.entry_point("main").unwrap(),
            ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        let vs_image = image_vs::load(self.device.clone()).unwrap();
        let fs_image = image_fs::load(self.device.clone()).unwrap();
        let image_pipeline = create_pipeline(
            self.device.clone(),
            render_pass.clone(),
            vs_image.entry_point("main").unwrap(),
            fs_image.entry_point("main").unwrap(),
            ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        self.pipelines
            .insert(("circle".to_string(), circle_pipeline));
        self.pipelines
            .insert(("square".to_string(), square_pipeline));
        self.pipelines.insert(("image".to_string(), image_pipeline));

        self.scheduler.0.update();

        let mut world_write = self.game.world.write().unwrap();
        let world = &mut world_write;

        let game_buffer = &mut self.game.world_buffer;

        for object in world.query_mut::<&mut Canvas>() {
            object.start(game_buffer);
        }

        game_buffer.execute_commands(world);

        for object in world.query_mut::<&mut DynObject>() {
            object.start(game_buffer);
        }

        game_buffer.execute_commands(world);

        drop(world_write);

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            viewport,
            recreate_swapchain,
            previous_frame_end,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.game.frames += 1;
        let rcx = self.rcx.as_mut().unwrap();
        self.scheduler.0.update();

        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.game.mouse_position = Some(position);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                let _span = tracy_client::span!("Engine::resize");
                rcx.recreate_swapchain = true;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.key_without_modifiers().as_ref() {
                        Key::Named(NamedKey::F2) => {
                            debug!("!!! DEBUG INFORMATION START !!!");
                            debug!(
                                pipelines_count = self.pipelines.len(),
                                descriptor_sets_count = self.descriptors.len(),
                                objects_count = self.game.world.read().unwrap().len(),
                                frame_count = self.game.frames
                            );
                            debug!("!!! DEBUG INFORMATION END !!!");
                        }
                        //#[cfg(debug_assertions)]
                        //Key::Named(NamedKey::F1) => {
                        //    debug!("Drawable calculation positions started");
                        //    if let Some(cursor) = self.game.mouse_position {
                        //        let drawables = self.game.children.get_by_position(cursor);
                        //        for drawable in drawables {
                        //            dbg!(&drawable);
                        //            debug!(
                        //                "drawable with id: {}",
                        //                drawable.lock().unwrap().render.mesh.get_id()
                        //            );
                        //        }
                        //    }
                        //    debug!("Drawable calculation positions finished");
                        //}
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let _span = tracy_client::span!("Engine::update");
                let window_size = rcx.window.inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // It is important to call this function from time to time, otherwise resources
                // will keep accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU
                // has already processed, and frees the resources that are no longer needed.
                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if rcx.recreate_swapchain {
                    // Use the new dimensions of the window.

                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;

                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass);

                    rcx.viewport.extent = window_size.into();

                    rcx.recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the
                // swapchain. If no image is available (which happens if you submit draw commands
                // too quickly), then the function will block. This operation returns the index of
                // the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional
                // timeout after which the function call will return an error.

                let span_acquire = tracy_client::span!("GPU: Acquire Next Image");
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                drop(span_acquire);

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let graphics_queue = self.queues.first().expect("Graphics queue not found");

                // In order to draw, we have to record a *command buffer*. The command buffer
                // object holds the list of commands that are going to be executed.
                //
                // Recording a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.

                let span_cmd = tracy_client::span!("GPU: Record Command Buffer");
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.memory.command_buffer_allocator.clone(),
                    graphics_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*.
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // A list of values to clear the attachments with. This list contains
                            // one item for each attachment in the render pass. In this case, there
                            // is only one attachment, and we clear it with a blue color.
                            //
                            // Only attachments that have `AttachmentLoadOp::Clear` are provided
                            // with clear values, any others should use `None` as the clear value.
                            clear_values: vec![Some([0.0, 0.0, 0.0, 0.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            // The contents of the first (and only) subpass. This can be either
                            // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                            // advanced and is not covered here.
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap();

                let (mesh_buffers, _children_size) =
                    calculate_drawables(self.memory.memory_allocator.clone(), &mut self.game, rcx);

                if let Some(mesh) = mesh_buffers {
                    let _span_draw =
                        tracy_client::span!("Engine:: Preparing Objects for Rendering");
                    builder.bind_vertex_buffers(0, mesh.0).unwrap();

                    let mut world = self.game.world.write().unwrap();

                    for (id, (class, shape, entity, color)) in world
                        .query_mut::<(hecs::Entity, (&ClassInfo, &Shapes, &mut DynObject, &Rgba8))>(
                        )
                    {
                        let _span_draw = tracy_client::span!("Engine: Draw Item");
                        let id = id.id() as usize - 1;
                        entity.update(&mut self.game.world_buffer);

                        let matrix = mesh.1[id];
                        let colour = color;
                        let constants = Constants(
                            matrix,
                            rcx.window.inner_size().into(),
                            (colour.r as u32)
                                | (colour.g as u32) << 8
                                | (colour.b as u32) << 16
                                | (colour.a as u32) << 24,
                        );

                        let pipeline = {
                            let shape_name = shape.as_ref().to_lowercase();

                            self.pipelines.get(&shape_name).expect("pipeline not found")
                        };

                        let layout = pipeline.layout();
                        if !layout.push_constant_ranges().is_empty() {
                            builder
                                .push_constants(pipeline.layout().clone(), 0, constants)
                                .unwrap();
                        }

                        let vertex_cursor = mesh.2[id];
                        let vertex_count = SQUARE_VERTEX.len() as u32;

                        builder.bind_pipeline_graphics(pipeline.clone()).unwrap();

                        if let Some(desc) = self.descriptors.get(&class.class_name) {
                            let _span_draw = tracy_client::span!("Engine: Getting descriptors");
                            builder
                                .bind_descriptor_sets(
                                    vulkano::pipeline::PipelineBindPoint::Graphics,
                                    pipeline.layout().clone(),
                                    0,
                                    desc.clone(),
                                )
                                .unwrap();
                        }

                        unsafe {
                            builder.draw(vertex_count, 1, vertex_cursor, 0).unwrap();
                        }
                    }
                    self.game.world_buffer.buffer.run_on(&mut world);
                }

                builder
                    // We leave the render pass. Note that if we had multiple subpasses we could
                    // have called `next_subpass` to jump to the next subpass.
                    .end_render_pass(Default::default())
                    .unwrap();

                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.build().unwrap();

                drop(span_cmd);

                let span_submit = tracy_client::span!("GPU: Submit & Present");
                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(graphics_queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        graphics_queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(&device).boxed());
                    }
                }

                drop(span_submit);
                tracy_client::Client::running().unwrap().frame_mark();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex, Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct SnakeVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
struct Constants(Transform, [f32; 2], u32);

pub trait RenderGameObject {
    fn shape(&self) -> Shapes;
}

pub trait RenderText {
    fn info(&self) -> SpriteTextCreateInfo;
}

/// Calculates Vertex buffer, matrices vector and offsets vector for draw in Vulkano
///
/// # Returns
/// tuple with buffer for vertices, matrices, offsets vectors
pub(crate) fn calculate_drawables(
    memory_allocator: Arc<dyn MemoryAllocator>,
    game: &mut GameContext,
    _rcx: &mut RenderContext,
) -> (Option<MeshBuffers>, usize) {
    let mut world_lock = game.world.write().unwrap();
    let entities_count = world_lock.len() as usize;
    if entities_count < 1 {
        return (None, entities_count);
    }
    let _span = tracy_client::span!("Engine::calculate_drawables");

    // Predicting the possible vector size
    let mut vertices: Vec<SnakeVertex> = Vec::with_capacity(entities_count * 2);
    let mut matrices: Vec<Transform> = Vec::with_capacity(entities_count);
    let mut offsets: Vec<u32> = Vec::with_capacity(entities_count);

    let drawable_size: usize = entities_count;

    for transform in world_lock.query_mut::<&Transform>() {
        let verts = &SQUARE_VERTEX;
        let matrix = transform;
        let offset = vertices.len() as u32;

        offsets.push(offset);
        vertices.extend_from_slice(verts);
        matrices.push(matrix.clone());
    }

    //debug!(
    //    vertices_size = vertices.len(),
    //    world_len = world_lock.world.len()
    //);

    let vertex_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    (
        Some(MeshBuffers(vertex_buffer, matrices, offsets)),
        drawable_size,
    )
}
