use std::sync::Arc;

use tracing::{debug, info};
use vulkano::{
    VulkanLibrary,
    buffer::BufferContents,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo,
        CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType,
        CommandBufferUsage, allocator::StandardCommandBufferAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexBuffersCollection, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    swapchain::Surface,
};
use winit::raw_window_handle::HasDisplayHandle;

use crate::{
    SnakeVertex,
    res::cache::{CacheProvider, DescriptorSetCache, PipelineCache}, shaders::{circle_shader::{circle_fs, circle_vs}, image_shader::{image_fs, image_vs}, square_shader::{square_fs, square_vs}},
};

pub(crate) struct SecondaryBufferBuilderCreateInfo<Pc> {
    pub shape_name: String,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub pipeline_cache: Arc<PipelineCache>,
    pub descriptor_set_cache: Arc<DescriptorSetCache>,
    pub descriptor_key: String,
    pub constants: Pc,
    pub vertex_count: u32,
    pub vertex_cursor: u32,
    pub framebuffer: Arc<Framebuffer>,
}

pub fn select_render_device(
    instance: Arc<Instance>,
    device_extensions: DeviceExtensions,
    event_loop: &impl HasDisplayHandle,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    let _span = tracy_client::span!("VULKAN: Selecting a device");
    info!("Selecting physical device (GPU)");
    // We then choose which physical device to use. First, we enumerate all the available
    // physical devices, then apply filters to narrow them down to those that can support our
    // needs.
    let (physical_device, graphics_index, transfer_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            // Some devices may not support the extensions or features that your application,
            // or report properties and limits that are not sufficient for your application.
            // These should be filtered out here.
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // For each physical device, we try to find a suitable queue family that will
            // execute our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a
            // draw queue and a compute queue), similar to CPU threads. This is
            // something you have to have to manage manually in Vulkan. Queues
            // of the same type belong to the same queue family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-world application, you may want to use a separate dedicated transfer queue
            // to handle data transfers in parallel with graphics operations.
            // You may also need a separate queue for compute operations, if
            // your application uses those.
            let family = p.queue_family_properties();
            let graphics = family
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // We select a queue family that supports graphics operations. When drawing
                    // to a window surface, as we do in this example, we also need to check
                    // that queues in this queue family are capable of presenting images to the
                    // surface.
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop)
                            .unwrap_or(false)
                })
                // The code here searches for the first queue family that is suitable. If none
                // is found, `None` is returned to `filter_map`, which
                // disqualifies this physical device.
                .map(|i| i as u32);

            let transfer = family
                .iter()
                .enumerate()
                .position(|(_i, q)| {
                    // We select a queue family that supports graphics operations. When drawing
                    // to a window surface, as we do in this example, we also need to check
                    // that queues in this queue family are capable of presenting images to the
                    // surface.
                    q.queue_flags.intersects(QueueFlags::TRANSFER)
                        && !q.queue_flags.contains(QueueFlags::GRAPHICS)
                })
                // The code here searches for the first queue family that is suitable. If none
                // is found, `None` is returned to `filter_map`, which
                // disqualifies this physical device.
                .map(|i| i as u32)
                .or(graphics);
            match (graphics, transfer) {
                (Some(g), Some(t)) => Some((p, g, t)),
                _ => panic!("Graphics device not found"),
            }
        })
        // All the physical devices that pass the filters above are suitable for the
        // application. However, not every device is equal, some are preferred over others.
        // Now, we assign each physical device a score, and pick the device with the lowest
        // ("best") score.
        //
        // In this example, we simply select the best-scoring device to use in the application.
        // In a real-world setting, you may want to use the best-scoring device only as a
        // "default" or "recommended" device, and let the user choose the device themself.
        .min_by_key(|(p, _, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");

    info!(
        device_name = ?physical_device.properties().device_name,
        device_type = ?physical_device.properties().device_type,
        max_push_constants_size = physical_device.properties().max_push_constants_size,
        "Selected physical device"
    );

    // Now initializing the device. This is probably the most important object of Vulkan.
    //
    // An iterator of created queues is returned by the function alongside the device.
    let (device, queues) = Device::new(
        // Which physical device to connect to.
        physical_device.clone(),
        DeviceCreateInfo {
            // A list of optional features and extensions that our program needs to work
            // correctly. Some parts of the Vulkan specs are optional and must be enabled
            // manually at device creation. In this example the only thing we are going to need
            // is the `khr_swapchain` extension that allows us to draw to a window.
            enabled_extensions: device_extensions.clone(),

            queue_create_infos: vec![
                QueueCreateInfo {
                    queue_family_index: graphics_index,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: transfer_index,
                    ..Default::default()
                },
            ],

            ..Default::default()
        },
    )
    .unwrap();

    // TODO: in the future I must create async compute ability
    for (i, queue) in device
        .physical_device()
        .queue_family_properties()
        .iter()
        .enumerate()
    {
        debug!(support_queues = ?queue.queue_flags, index = ?i);
    }

    (device, queues)
}
// Before we draw, we have to create what is called a **pipeline**. A pipeline describes
// how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
// contains many settings for customization, all baked into a single object. For drawing,
// we create a **graphics** pipeline, but there are also other types of pipeline.
pub fn create_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    vs: vulkano::shader::EntryPoint,
    fs: vulkano::shader::EntryPoint,
    blend_state: ColorBlendState,
) -> Arc<GraphicsPipeline> {
    let _span = tracy_client::span!("Engine: Creating pipeline");
    let vertex_input_state = SnakeVertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    let subpass = Subpass::from(render_pass, 0).unwrap();

    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::TriangleFan,
                ..Default::default()
            }),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(blend_state),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(crate) fn get_vulkan_instance(
    event_loop: &impl HasDisplayHandle,
    mut extensions: Vec<String>,
) -> Arc<Instance> {
    debug!("Initializing Vulkan library");
    let library = VulkanLibrary::new()
        .expect("Vulkan not found. You may not have Vulkan support or an up-to-date GPU driver.");

    debug!("Gathering required Vulkan extensions for windowing");

    let mut required_extensions = Surface::required_extensions(event_loop).unwrap();
    required_extensions.ext_debug_utils = true;
    let supported_extensions = library.supported_extensions();

    for extension in supported_extensions.clone().into_iter().filter(|e| e.1) {
        debug!("Supported extension: {}", extension.0);
    }

    required_extensions &= *supported_extensions;

    for enabled_extension in required_extensions.clone().into_iter().filter(|e| e.1) {
        debug!("Enabled extension: {}", enabled_extension.0);
    }

    let mut enabled_layers: Vec<String> = Vec::new();
    enabled_layers.append(&mut extensions);

    #[cfg(debug_assertions)]
    {
        let validation_extension = String::from("VK_LAYER_KHRONOS_validation");
        if !enabled_layers.contains(&validation_extension) {
            enabled_layers.push(validation_extension);
        }
    }

    for layer in enabled_layers.iter() {
        debug!("Enabled layer: {}", layer);
    }

    debug!("Creating Vulkan instance");
    let instance = Instance::new(
        library.clone(),
        InstanceCreateInfo {
            // Enable enumerating devices that use non-conformant Vulkan implementations.
            // (e.g. MoltenVK)
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            enabled_layers: enabled_layers,
            ..Default::default()
        },
    )
    .unwrap();

    instance
}

pub(crate) fn secondary_buffer_create<Pc>(
    cmd: SecondaryBufferBuilderCreateInfo<Pc>,
    vertex_buffer: impl VertexBuffersCollection,
    viewport: Viewport,
    subpass: Subpass,
    queue_index: u32,
) -> Result<
    Arc<vulkano::command_buffer::SecondaryAutoCommandBuffer>,
    vulkano::Validated<vulkano::VulkanError>,
>
where
    Pc: BufferContents,
{
    let command_buffer_allocator = cmd.command_buffer_allocator.clone();
    let pipeline_cache = cmd.pipeline_cache.clone();
    let descriptor_set_cache = cmd.descriptor_set_cache.clone();

    let mut builder = AutoCommandBufferBuilder::secondary(
        command_buffer_allocator,
        queue_index,
        CommandBufferUsage::OneTimeSubmit,
        CommandBufferInheritanceInfo {
            render_pass: Some(CommandBufferInheritanceRenderPassType::BeginRenderPass(
                CommandBufferInheritanceRenderPassInfo {
                    subpass,
                    framebuffer: Some(cmd.framebuffer.clone()),
                },
            )),
            ..Default::default()
        },
    )
    .expect("Secondary buffer creating error");

    let pipeline = pipeline_cache.get(&cmd.shape_name).unwrap();

    let layout = pipeline.layout();
    if !layout.push_constant_ranges().is_empty() {
        builder
            .push_constants(pipeline.layout().clone(), 0, cmd.constants)
            .unwrap();
    }

    builder.bind_pipeline_graphics(pipeline.clone()).unwrap();
    if let Some(desc) = descriptor_set_cache.get(&cmd.descriptor_key) {
        builder
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                desc.clone(),
            )
            .unwrap();
    }

    builder
        .bind_pipeline_graphics(pipeline.clone())
        .unwrap()
        .set_viewport(0, [viewport].into_iter().collect())
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer)
        .unwrap();

    unsafe {
        builder
            .draw(cmd.vertex_count, 1, cmd.vertex_cursor, 0)
            .unwrap();
    }

    builder.build()
}



pub(crate) fn add_shaders(
    pipelines: Arc<PipelineCache>,
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
) {
    let vs_square = square_vs::load(device.clone()).unwrap();
    let fs_square = square_fs::load(device.clone()).unwrap();
    let square_pipeline = create_pipeline(
        device.clone(),
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

    let vs_circle = circle_vs::load(device.clone()).unwrap();
    let fs_circle = circle_fs::load(device.clone()).unwrap();
    let circle_pipeline = create_pipeline(
        device.clone(),
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

    let vs_image = image_vs::load(device.clone()).unwrap();
    let fs_image = image_fs::load(device.clone()).unwrap();
    let image_pipeline = create_pipeline(
        device.clone(),
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

    pipelines.insert(("circle".to_string(), circle_pipeline));
    pipelines.insert(("square".to_string(), square_pipeline));
    pipelines.insert(("image".to_string(), image_pipeline));
}
