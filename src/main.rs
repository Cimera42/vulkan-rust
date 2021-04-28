// fn main() {
//     println!("Hello, world!");
// }

extern crate glfw;

use glfw::{Action, Context, Key};
use serde::{Deserialize, Serialize};
use std::fs;
use std::mem::ManuallyDrop;
use gfx_hal::{
    device::Device,
    window::{Extent2D, PresentationSurface, Surface},
    Instance,
};
use shaderc::ShaderKind;

mod shaders;

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
struct Config {
    window: Window,
    shaders: ShaderPaths,
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
            } = ManuallyDrop::take(&mut self.0);

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
        BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face, GraphicsPipelineDesc,
        InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer, Specialization,
    };

    let vertex_shader_module = shaders::create_module::<B>(device, vertex_shader_filename, ShaderKind::Vertex);
    let fragment_shader_module = shaders::create_module::<B>(device, fragment_shader_filename, ShaderKind::Fragment);

    let (vs_entry, fs_entry) = (
        EntryPoint {
            entry: "main",
            module: &vertex_shader_module,
            specialization: Specialization::default(),
        },
        EntryPoint {
            entry: "main",
            module: &fragment_shader_module,
            specialization: Specialization::default(),
        },
    );

    let primitive_assembler = PrimitiveAssemblerDesc::Vertex {
        buffers: &[],
        attributes: &[],
        input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
        vertex: vs_entry,
        tessellation: None,
        geometry: None,
    };

    let mut pipeline_desc = GraphicsPipelineDesc::new(
        primitive_assembler,
        Rasterizer {
            cull_face: Face::BACK,
            ..Rasterizer::FILL
        },
        Some(fs_entry),
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

    let pipeline = device
            .create_graphics_pipeline(&pipeline_desc, None)
            .expect("Failed to create graphics pipeline");

    device.destroy_shader_module(vertex_shader_module);
    device.destroy_shader_module(fragment_shader_module);

    return pipeline;
}

fn main() {
    let config = load_config("assets/config.json");

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (logical_window_size, physical_window_size) =
        glfw.with_primary_monitor(|_, m| {
            let monitor = m.expect("No monitor?");
            let (xscale, yscale) = monitor.get_content_scale();
            println!("{0} {1}", xscale, yscale);
            let logical =  [config.window.width, config.window.height];
            let physical = [((logical[0] as f32) * xscale) as u32, ((logical[1] as f32) * yscale) as u32];

            (logical, physical)
        });

    let (mut window, events) = glfw.create_window(logical_window_size[0], logical_window_size[1], &config.window.name, glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.set_key_polling(true);
    window.make_current();

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

    let (command_pool, mut command_buffer) = unsafe {
        use gfx_hal::command::Level;
        use gfx_hal::pool::{CommandPool, CommandPoolCreateFlags};

        let mut command_pool = device
            .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
            .expect("Out of memory");

        let command_buffer = command_pool.allocate_one(Level::Primary);

        (command_pool, command_buffer)
    };

    let surface_color_format = {
        use gfx_hal::format::{ChannelType, Format};

        let supported_formats = surface
            .supported_formats(&adapter.physical_device)
            .unwrap_or(vec![]);

        let default_format = *supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);

        supported_formats
            .into_iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .unwrap_or(default_format)
    };

    let render_pass = {
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

        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        unsafe {
            device
                .create_render_pass(&[color_attachment], &[subpass], &[])
                .expect("Out of memory")
        }
    };

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&[], &[])
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
        }));

    let mut should_configure_swapchain = true;

    window.make_current();

    while !window.should_close() {
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                glfw::WindowEvent::Size(newWidth, newHeight) => {
                    surface_extent = Extent2D {
                        width: newWidth as u32,
                        height: newHeight as u32,
                    };
                    should_configure_swapchain = true;
                }
                _ => (),
            }
        }

        let res: &mut Resources<_> = &mut resource_holder.0;
        let render_pass = &res.render_passes[0];
        let pipeline = &res.pipelines[0];

        unsafe {
            use gfx_hal::pool::CommandPool;

            // We refuse to wait more than a second, to avoid hanging.
            let render_timeout_ns = 1_000_000_000;

            res.device
                .wait_for_fence(&res.submission_complete_fence, render_timeout_ns)
                .expect("Out of memory or device lost");

            res.device
                .reset_fence(&res.submission_complete_fence)
                .expect("Out of memory");

            res.command_pool.reset(false);
        }

        if should_configure_swapchain {
            use gfx_hal::window::SwapchainConfig;

            let caps = res.surface.capabilities(&adapter.physical_device);

            let mut swapchain_config =
                SwapchainConfig::from_caps(&caps, surface_color_format, surface_extent);

            // This seems to fix some fullscreen slowdown on macOS.
            if caps.image_count.contains(&3) {
                swapchain_config.image_count = 3;
            }

            surface_extent = swapchain_config.extent;

            unsafe {
                res.surface
                    .configure_swapchain(&res.device, swapchain_config)
                    .expect("Failed to configure swapchain");
            };

            should_configure_swapchain = false;
        }

        let surface_image = unsafe {
            // We refuse to wait more than a second, to avoid hanging.
            let acquire_timeout_ns = 1_000_000_000;

            match res.surface.acquire_image(acquire_timeout_ns) {
                Ok((image, _)) => image,
                Err(_) => {
                    should_configure_swapchain = true;
                    return;
                }
            }
        };

        let framebuffer = unsafe {
            use std::borrow::Borrow;

            use gfx_hal::image::Extent;

            res.device
                .create_framebuffer(
                    render_pass,
                    vec![surface_image.borrow()],
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

            command_buffer.begin_render_pass(
                render_pass,
                &framebuffer,
                viewport.rect,
                &[ClearValue {
                    color: ClearColor {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }],
                SubpassContents::Inline,
            );

            command_buffer.bind_graphics_pipeline(pipeline);

            command_buffer.draw(0..3, 0..1);
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
