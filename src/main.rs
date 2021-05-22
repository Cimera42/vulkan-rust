// fn main() {
//     println!("Hello, world!");
// }

extern crate glfw;

use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Rotation3};
use glfw::{Action, Context, Key};
use serde::{Deserialize, Serialize};
use std::{fs, mem::{ManuallyDrop}};
use gfx_hal::{Backend, Instance, adapter::Adapter, buffer::{IndexBufferView, SubRange}, command::ClearDepthStencil, device::Device, format::{Aspects, D32Sfloat, Format, Swizzle}, image::{Access, Kind, Layout, SubresourceRange, Usage, ViewCapabilities}, memory::Dependencies, pass::{SubpassDependency, SubpassId}, pso::{DepthStencilDesc, DepthTest, PipelineStage}, window::{Extent2D, PresentationSurface, Surface}};
use shaderc::ShaderKind;

mod shaders;
mod texture;

#[derive(Serialize, Deserialize)]
struct Window {
    name: String,
    width: u32,
    height: u32,
}

#[derive(Serialize, Deserialize)]
struct ShaderPaths {
    vertex: String,
    fragment: String,
}

#[derive(Serialize, Deserialize)]
struct Node {
    path: String,
}

#[derive(Serialize, Deserialize)]
struct Scene {
    nodes: Vec<Node>,
}


#[derive(Serialize, Deserialize)]
struct Config {
    window: Window,
    shaders: ShaderPaths,
    scene: Scene,
}

struct Resources<B: gfx_hal::Backend> {
    instance: B::Instance,
    surface: B::Surface,
    device: B::Device,
    render_passes: Vec<B::RenderPass>,
    pipeline_layouts: Vec<B::PipelineLayout>,
    pipelines: Vec<B::GraphicsPipeline>,
    command_pool: B::CommandPool,
    submission_complete_fence: B::Fence,
    rendering_complete_semaphore: B::Semaphore,

    vertex_buffer_memory: B::Memory,
    vertex_buffer: B::Buffer,

    index_buffer_memory: B::Memory,
    index_buffer: B::Buffer,
}

struct ResourceHolder<B: gfx_hal::Backend>(ManuallyDrop<Resources<B>>);


impl<B: gfx_hal::Backend> Drop for ResourceHolder<B> {
    fn drop(&mut self) {
        unsafe {
            let Resources {
                instance,
                mut surface,
                device,
                command_pool,
                render_passes,
                pipeline_layouts,
                pipelines,
                submission_complete_fence,
                rendering_complete_semaphore,
                vertex_buffer_memory,
                vertex_buffer,
                index_buffer_memory,
                index_buffer,
            } = ManuallyDrop::take(&mut self.0);

            device.free_memory(vertex_buffer_memory);
            device.destroy_buffer(vertex_buffer);

            device.free_memory(index_buffer_memory);
            device.destroy_buffer(index_buffer);

            device.destroy_semaphore(rendering_complete_semaphore);
            device.destroy_fence(submission_complete_fence);
            for pipeline in pipelines {
                device.destroy_graphics_pipeline(pipeline);
            }
            for pipeline_layout in pipeline_layouts {
                device.destroy_pipeline_layout(pipeline_layout);
            }
            for render_pass in render_passes {
                device.destroy_render_pass(render_pass);
            }
            device.destroy_command_pool(command_pool);
            surface.unconfigure_swapchain(&device);
            instance.destroy_surface(surface);
        }
    }
}

#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PushConstants {
    // model_matrix: Matrix4<f32>,
    // view_matrix: Matrix4<f32>,
    // projection_matrix: Matrix4<f32>,
    mvp_matrix: Matrix4<f32>,
}


fn load_config(filename: &str) -> Config {
    let config_string  = fs::read_to_string(filename).expect("Could not load config file");
    let config: Config = serde_json::from_str(&config_string).expect("JSON was not well-formatted");
    return config;
}

unsafe fn make_pipeline<B: gfx_hal::Backend>(
    device: &B::Device,
    render_pass: &B::RenderPass,
    pipeline_layout: &B::PipelineLayout,
    vertex_shader_filename: &str,
    fragment_shader_filename: &str,
) -> B::GraphicsPipeline {
    use gfx_hal::pass::Subpass;
    use gfx_hal::pso::{
        BlendState, ColorBlendDesc, ColorMask, Face, GraphicsPipelineDesc,
        InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer,
    };

    let vertex_shader = shaders::create_module::<B>(device, vertex_shader_filename, ShaderKind::Vertex);
    let vertex_entry = shaders::create_entry(&vertex_shader);
    let fragment_shader = shaders::create_module::<B>(device, fragment_shader_filename, ShaderKind::Fragment);
    let frag_entry = shaders::create_entry(&fragment_shader);

    let primitive_assembler = {
        use gfx_hal::pso::{AttributeDesc, Element, VertexBufferDesc, VertexInputRate};

        PrimitiveAssemblerDesc::Vertex {
            buffers: &[VertexBufferDesc {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: VertexInputRate::Vertex,
            }],

            attributes: &[
                AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: Element {
                        format: Format::Rgb32Sfloat,
                        offset: 0,
                    },
                },
                AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: Element {
                        format: Format::Rgb32Sfloat,
                        offset: 12,
                    },
                },
            ],
            input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
            vertex: vertex_entry,
            tessellation: None,
            geometry: None,
        }
    };

    let mut pipeline_desc = GraphicsPipelineDesc::new(
        primitive_assembler,
        Rasterizer {
            cull_face: Face::BACK,
            front_face: gfx_hal::pso::FrontFace::Clockwise,
            ..Rasterizer::FILL
        },
        Some(frag_entry),
        pipeline_layout,
        Subpass {
            index: 0,
            main_pass: render_pass,
        },
    );

    pipeline_desc.blender.targets.push(ColorBlendDesc {
        mask: ColorMask::ALL,
        blend: Some(BlendState::ALPHA),
    });

    pipeline_desc.depth_stencil = DepthStencilDesc {
        depth: Some(DepthTest {
            fun: gfx_hal::pso::Comparison::Less,
            write: true,
        }),
        ..Default::default()
    };

    let pipeline = device
            .create_graphics_pipeline(&pipeline_desc, None)
            .expect("Failed to create graphics pipeline");

    device.destroy_shader_module(vertex_shader);
    device.destroy_shader_module(fragment_shader);

    return pipeline;
}


fn make_render_pass<B: gfx_hal::Backend>(device: &B::Device, surface_color_format: Format, surface_depth_format: Format) -> B::RenderPass {
    use gfx_hal::image::Layout;
    use gfx_hal::pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc,
    };

    let color_attachment = Attachment {
        format: Some(surface_color_format),
        samples: 1,
        ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
        stencil_ops: AttachmentOps::DONT_CARE,
        layouts: Layout::Undefined..Layout::Present,
    };

    let depth_attachment = Attachment {
        format: Some(Format::D32SfloatS8Uint/* surface_depth_format */),
        samples: 1,
        ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::DontCare),
        stencil_ops: AttachmentOps::DONT_CARE,
        layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
    };

    let subpass = SubpassDesc {
        colors: &[(0, Layout::ColorAttachmentOptimal)],
        depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
        inputs: &[],
        resolves: &[],
        preserves: &[],
    };

    // let dependency = SubpassDependency {
    //     passes: None..Some(0),
    //     stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..
    //         PipelineStage::COLOR_ATTACHMENT_OUTPUT,
    //     accesses: Access::empty()..
    //         (Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE),
    //     flags: Dependencies::empty(),
    // };

    return unsafe {
        device
            .create_render_pass(&[color_attachment, depth_attachment], &[subpass], &[/* dependency */])
            .expect("Out of memory")
    }
}


unsafe fn wait_for_fence<B: gfx_hal::Backend>(device: &B::Device, fence: &B::Fence) {
    // We refuse to wait more than a second, to avoid hanging.
    let render_timeout_ns = 1_000_000_000;

    device
        .wait_for_fence(fence, render_timeout_ns)
        .expect("Out of memory or device lost");

    device
        .reset_fence(fence)
        .expect("Out of memory");

}

fn reconfigure_swapchain<B: gfx_hal::Backend>(device: &B::Device, surface: &mut B::Surface, adapter: &Adapter<B>, surface_color_format: Format, surface_extent: &mut Extent2D) {
    use gfx_hal::window::SwapchainConfig;

    let caps = surface.capabilities(&adapter.physical_device);

    let mut swapchain_config =
        SwapchainConfig::from_caps(&caps, surface_color_format, *surface_extent);

    // This seems to fix some fullscreen slowdown on macOS.
    if caps.image_count.contains(&3) {
        swapchain_config.image_count = 3;
    }

    *surface_extent = swapchain_config.extent;

    unsafe {
        surface
            .configure_swapchain(device, swapchain_config)
            .expect("Failed to configure swapchain");
    };
}

unsafe fn make_buffer<B: gfx_hal::Backend>(
    device: &B::Device,
    physical_device: &B::PhysicalDevice,
    buffer_len: usize,
    usage: gfx_hal::buffer::Usage,
    properties: gfx_hal::memory::Properties,
) -> (B::Memory, B::Buffer) {
    use gfx_hal::{adapter::PhysicalDevice, MemoryTypeId};

    let mut buffer = device
        .create_buffer(buffer_len as u64, usage)
        .expect("Failed to create buffer");

    let req = device.get_buffer_requirements(&buffer);

    let memory_types = physical_device.memory_properties().memory_types;

    let memory_type = memory_types
        .iter()
        .enumerate()
        .find(|(id, mem_type)| {
            let type_supported = req.type_mask & (1_u32 << id) != 0;
            type_supported && mem_type.properties.contains(properties)
        })
        .map(|(id, _ty)| MemoryTypeId(id))
        .expect("No compatible memory type available");

    let buffer_memory = device
        .allocate_memory(memory_type, req.size)
        .expect("Failed to allocate buffer memory");

    device
        .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
        .expect("Failed to bind buffer memory");

    (buffer_memory, buffer)
}

fn read_mesh(filename: &str) -> Option<Mesh> {
    let (gltf, buffers, _images) = gltf::import(filename).expect("Could not load model");
    for gltf_mesh in gltf.meshes() {
        println!("Mesh #{}", gltf_mesh.index());
        for primitive in gltf_mesh.primitives() {
            println!("- Primitive #{}", primitive.index());
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions = reader.read_positions().unwrap().collect::<Vec<_>>();
            let normals = reader.read_normals().unwrap().collect::<Vec<_>>();
            let vertices = positions.into_iter().enumerate().map(|(index, position)| {
                Vertex {
                    position: position,
                    normal: normals[index],
                }
            }).collect::<Vec<Vertex>>();

            let indices = reader.read_indices().expect("Mesh does not have indices").into_u32().collect::<Vec<_>>();

            return Option::Some(Mesh {
                vertices,
                indices,
            })
        }
    }
    return Option::None;
}

unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
    let size_in_bytes = std::mem::size_of::<T>();
    let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
    let start_ptr = push_constants as *const T as *const u32;
    std::slice::from_raw_parts(start_ptr, size_in_u32s)
}

fn vec_size<T>(vector: &Vec<T>) -> usize {
    return vector.len() * std::mem::size_of::<T>();
}

fn main() {
    let config = load_config("assets/config.json");

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (logical_window_size, physical_window_size) =
        glfw.with_primary_monitor(|_, m| {
            let monitor = m.expect("No monitor?");
            let (xscale, yscale) = monitor.get_content_scale();

            let logical =  [config.window.width, config.window.height];
            let physical = [((logical[0] as f32) * xscale) as u32, ((logical[1] as f32) * yscale) as u32];

            (logical, physical)
        });

    let (mut window, events) = glfw.create_window(logical_window_size[0], logical_window_size[1], &config.window.name, glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.set_key_polling(true);
    window.set_size_polling(true);
    window.set_cursor_mode(glfw::CursorMode::Disabled);
    window.set_cursor_pos(0.0, 0.0);
    window.make_current();

    let mesh = read_mesh(&config.scene.nodes[0].path).expect("Mesh not initialized");

    let mut surface_extent = Extent2D {
        width: physical_window_size[0],
        height: physical_window_size[1],
    };

    let (instance, surface, adapter) = {
        let instance = backend::Instance::create(&config.window.name, 1).expect("Backend not supported");

        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create surface for window")
        };

        let adapter = instance.enumerate_adapters().remove(0);

        (instance, surface, adapter)
    };

    let (device, mut queue_group) = {
        use gfx_hal::queue::QueueFamily;

        let queue_family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .expect("No compatible queue family found");

        let mut gpu = unsafe {
            use gfx_hal::adapter::PhysicalDevice;

            adapter
                .physical_device
                .open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
                .expect("Failed to open device")
        };

        (gpu.device, gpu.queue_groups.pop().unwrap())
    };

    let vertex_buffer_len = vec_size(&mesh.vertices);

    let (vertex_buffer_memory, vertex_buffer) = unsafe {
        use gfx_hal::buffer::Usage;
        use gfx_hal::memory::Properties;

        make_buffer::<backend::Backend>(
            &device,
            &adapter.physical_device,
            vertex_buffer_len,
            Usage::VERTEX,
            Properties::CPU_VISIBLE,
        )
    };

    unsafe {
        use gfx_hal::memory::Segment;

        let mapped_memory = device
            .map_memory(&vertex_buffer_memory, Segment::ALL)
            .expect("Failed to map memory");

        std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr() as *const u8, mapped_memory, vertex_buffer_len);

        device
            .flush_mapped_memory_ranges(vec![(&vertex_buffer_memory, Segment::ALL)])
            .expect("Out of memory");

        device.unmap_memory(&vertex_buffer_memory);
    }

    let index_buffer_len = vec_size(&mesh.indices);

    let (index_buffer_memory, index_buffer) = unsafe {
        use gfx_hal::buffer::Usage;
        use gfx_hal::memory::Properties;

        make_buffer::<backend::Backend>(
            &device,
            &adapter.physical_device,
            index_buffer_len,
            Usage::INDEX,
            Properties::CPU_VISIBLE,
        )
    };

    unsafe {
        use gfx_hal::memory::Segment;

        let mapped_memory = device
            .map_memory(&index_buffer_memory, Segment::ALL)
            .expect("Failed to map memory");

        std::ptr::copy_nonoverlapping(mesh.indices.as_ptr() as *const u8, mapped_memory, index_buffer_len);

        device
            .flush_mapped_memory_ranges(vec![(&index_buffer_memory, Segment::ALL)])
            .expect("Out of memory");

        device.unmap_memory(&index_buffer_memory);
    }


    let (command_pool, mut command_buffer) = unsafe {
        use gfx_hal::command::Level;
        use gfx_hal::pool::{CommandPool, CommandPoolCreateFlags};

        let mut command_pool = device
            .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
            .expect("Out of memory");

        let command_buffer = command_pool.allocate_one(Level::Primary);

        (command_pool, command_buffer)
    };

    let (surface_color_format, surface_depth_format) = {
        use gfx_hal::format::{ChannelType};

        let supported_formats = surface
            .supported_formats(&adapter.physical_device)
            .unwrap_or(vec![]);

        let default_format = supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);

        let color_format = (&supported_formats)
            .into_iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .unwrap_or(default_format);

        let default_depth_format = supported_formats.get(0).unwrap_or(&Format::D32Sfloat);

        let depth_format = (&supported_formats)
            .into_iter()
            .find(|format| format.base_format().1 == ChannelType::Sfloat)
            .unwrap_or(default_depth_format);

        (*color_format, *depth_format)
    };

    let render_pass = make_render_pass::<backend::Backend>(&device, surface_color_format, surface_depth_format);

    let pipeline_layout = unsafe {
        use gfx_hal::pso::ShaderStageFlags;

        let push_constant_bytes = std::mem::size_of::<PushConstants>() as u32;

        device
            .create_pipeline_layout(&[], &[(ShaderStageFlags::VERTEX, 0..push_constant_bytes)])
            .expect("Out of memory")
    };

    let pipeline = unsafe {
        make_pipeline::<backend::Backend>(
            &device,
            &render_pass,
            &pipeline_layout,
            &config.shaders.vertex,
            &config.shaders.fragment,
        )
    };


    let submission_complete_fence = device.create_fence(true).expect("Out of memory");
    let rendering_complete_semaphore = device.create_semaphore().expect("Out of memory");

    let mut resource_holder: ResourceHolder<backend::Backend> =
        ResourceHolder(ManuallyDrop::new(Resources {
            instance,
            surface,
            device,
            command_pool,
            render_passes: vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],
            pipelines: vec![pipeline],
            submission_complete_fence,
            rendering_complete_semaphore,
            vertex_buffer_memory,
            vertex_buffer,
            index_buffer_memory,
            index_buffer,
        }));

    let mut should_configure_swapchain = true;

    // let start_time = std::time::Instant::now();
    // let angle = start_time.elapsed().as_secs_f32();

    let mut camera_rotation = Quaternion::<f32>::new(1.0,0.0,0.0,0.0).normalize();
    let mut camera_position = cgmath::point3(0.0, 0.0, 0.0);

    while !window.should_close() {
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                glfw::WindowEvent::Size(new_width, new_height) => {
                    surface_extent = Extent2D {
                        width: new_width as u32,
                        height: new_height as u32,
                    };
                    should_configure_swapchain = true;
                }
                _ => (),
            }
        }

        let (x, y) = window.get_cursor_pos();
        let yaw: Quaternion<f32> = Rotation3::from_axis_angle(cgmath::vec3(0.0, -1.0, 0.0), cgmath::Deg(x as f32 / 2.0));
        let left = camera_rotation * cgmath::vec3(-1.0, 0.0, 0.0);
        let pitch: Quaternion<f32> = Rotation3::from_axis_angle(left, cgmath::Deg(y as f32 / 2.0));
        camera_rotation = yaw * pitch * camera_rotation;
        window.set_cursor_pos(0.0, 0.0);

        let new_right = camera_rotation * cgmath::vec3(-1.0, 0.0, 0.0) * 0.25;
        let new_forward = camera_rotation * cgmath::vec3(0.0, 0.0, 1.0) * 0.25;

        if window.get_key(Key::D) == Action::Press {
            camera_position += new_right;
        }

        if window.get_key(Key::A) == Action::Press {
            camera_position -= new_right;
        }

        if window.get_key(Key::W) == Action::Press {
            camera_position += new_forward;
        }

        if window.get_key(Key::S) == Action::Press {
            camera_position -= new_forward;
        }

        let res: &mut Resources<backend::Backend> = &mut resource_holder.0;
        let render_pass = &res.render_passes[0];
        let pipeline = &res.pipelines[0];
        let pipeline_layout = &res.pipeline_layouts[0];

        let model_matrix = Matrix4::from_translation(cgmath::vec3::<f32>(0.0, 0.0, 5.0));
        let view_matrix: Matrix4<f32> =Matrix4::look_at_lh(
            camera_position + (camera_rotation * cgmath::vec3(0.0, 0.0, 1.0)),
            camera_position,
            camera_rotation * cgmath::vec3(0.0, 1.0, 0.0),
        );
        let projection_matrix = cgmath::perspective(cgmath::Deg(100.0), config.window.width as f32 / config.window.height as f32, 0.1, 100.0);
        let push_constants = PushConstants {
            // model_matrix,
            // view_matrix,
            // projection_matrix,
            mvp_matrix: projection_matrix * view_matrix * model_matrix
        };

        unsafe {
            use gfx_hal::pool::CommandPool;

            wait_for_fence::<backend::Backend>(&res.device, &res.submission_complete_fence);

            res.command_pool.reset(false);
        }

        if should_configure_swapchain {
            reconfigure_swapchain(&res.device, &mut res.surface, &adapter, surface_color_format, &mut surface_extent);

            should_configure_swapchain = false;
        }

        let surface_image = unsafe {
            // We refuse to wait more than a second, to avoid hanging.
            let acquire_timeout_ns = 1_000_000_000;

            match res.surface.acquire_image(acquire_timeout_ns) {
                Ok((image, _)) => image,
                Err(_) => {
                    should_configure_swapchain = true;
                    continue;
                }
            }
        };

        let (depth_image, depth_image_memory) = unsafe {
            texture::create_image::<backend::Backend>(
                &res.device,
                &adapter.physical_device,
                texture::ImageInfo {
                    kind: Kind::D2(surface_extent.width, surface_extent.height, 1, 1),
                    format: Format::D32SfloatS8Uint,//surface_depth_format,
                    tiling: gfx_hal::image::Tiling::Optimal,
                    usage: Usage::DEPTH_STENCIL_ATTACHMENT,
                    view_caps: ViewCapabilities::empty()
                }
            )
        };

        let depth_image_view = unsafe {
            texture::create_image_view::<backend::Backend>(
                &res.device,
                &depth_image,
                texture::ImageViewInfo {
                    kind: gfx_hal::image::ViewKind::D2,
                    format: Format::D32SfloatS8Uint,//surface_depth_format,
                    aspects: Aspects::DEPTH,
                }
            )
        };

        unsafe {
            texture::transition_image_layout::<backend::Backend>(
                &mut res.command_pool,
                Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
                &depth_image,
                &mut queue_group.queues[0],
            );
        }

        let framebuffer = unsafe {
            use std::borrow::Borrow;

            use gfx_hal::image::Extent;

            res.device
                .create_framebuffer(
                    render_pass,
                    vec![surface_image.borrow(), depth_image_view.borrow()],
                    Extent {
                        width: surface_extent.width,
                        height: surface_extent.height,
                        depth: 1,
                    },
                )
                .unwrap()
        };

        let viewport = {
            use gfx_hal::pso::{Rect, Viewport};

            Viewport {
                rect: Rect {
                    x: 0,
                    y: 0,
                    w: surface_extent.width as i16,
                    h: surface_extent.height as i16,
                },
                depth: 0.0..1.0,
            }
        };

        unsafe {
            use gfx_hal::command::{
                ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, SubpassContents,
            };

            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

            command_buffer.set_viewports(0, &[viewport.clone()]);
            command_buffer.set_scissors(0, &[viewport.rect]);

            command_buffer.bind_vertex_buffers(
                0,
                vec![(&res.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)],
            );

            let indices_count = mesh.indices.len() as u32;
            command_buffer.bind_index_buffer(
                IndexBufferView {
                    buffer: &res.index_buffer,
                    index_type: gfx_hal::IndexType::U32,
                    range: SubRange {
                        offset: 0,
                        size: Option::None,
                    },
                }
            );

            command_buffer.begin_render_pass(
                render_pass,
                &framebuffer,
                viewport.rect,
                &[ClearValue {
                    color: ClearColor {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }, ClearValue {
                    depth_stencil: ClearDepthStencil {
                        depth: 1.0,
                        stencil: 0,
                    }
                }],
                SubpassContents::Inline,
            );

            command_buffer.bind_graphics_pipeline(pipeline);

            use gfx_hal::pso::ShaderStageFlags;

            command_buffer.push_graphics_constants(
                pipeline_layout,
                ShaderStageFlags::VERTEX,
                0,
                push_constant_bytes(&[push_constants]),
            );

            command_buffer.draw_indexed(0..indices_count, 0, 0..1);
            // command_buffer.draw(0..vertices_count, 0..1);

            command_buffer.end_render_pass();
            command_buffer.finish();
        }

        unsafe {
            use gfx_hal::queue::{CommandQueue, Submission};

            let submission = Submission {
                command_buffers: vec![&command_buffer],
                wait_semaphores: None,
                signal_semaphores: vec![&res.rendering_complete_semaphore],
            };

            queue_group.queues[0].submit(submission, Some(&res.submission_complete_fence));
            let result = queue_group.queues[0].present(
                &mut res.surface,
                surface_image,
                Some(&res.rendering_complete_semaphore),
            );

            should_configure_swapchain |= result.is_err();

            res.device.destroy_framebuffer(framebuffer);
        }

        window.swap_buffers();
        glfw.poll_events();
    }
}
