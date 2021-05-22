use std::{iter::once, mem, ops::Range, process::exit};

use gfx_hal::{Backend, MemoryTypeId, adapter::MemoryType, command::{CommandBuffer, CommandBufferFlags, Level}, device::Device, format::{Aspects, Swizzle}, image::{Access, Layout, SubresourceRange}, memory::{Barrier, Dependencies, Properties}, pool::CommandPool, prelude::{CommandQueue, PhysicalDevice}, pso::PipelineStage, queue::Submission};

pub struct ImageInfo {
    pub kind: gfx_hal::image::Kind,
    pub format: gfx_hal::format::Format,
    pub tiling: gfx_hal::image::Tiling,
    pub usage: gfx_hal::image::Usage,
    pub view_caps: gfx_hal::image::ViewCapabilities,
}

pub struct ImageViewInfo {
    pub kind: gfx_hal::image::ViewKind,
    pub format: gfx_hal::format::Format,
    pub aspects: gfx_hal::format::Aspects,
}

fn get_memory_type_id<B: gfx_hal::Backend>(physical_device: &B::PhysicalDevice, type_mask: u32, properties: Properties) -> Option<usize> {
    let memory_properties = physical_device.memory_properties();
    for (i, property) in memory_properties.memory_types.iter().enumerate() {
        if (type_mask & (1 << i) > 0) && property.properties.contains(properties) {
            return Some(i);
        }
    }
    return None;
}

pub unsafe fn create_image<B: gfx_hal::Backend>(device: &B::Device, physical_device: &B::PhysicalDevice, info: ImageInfo) -> (B::Image, B::Memory) {
    let mut image = device.create_image(
        info.kind,
        1,
        info.format,
        info.tiling,
        info.usage,
        info.view_caps,
    ).expect("Could not create image");

    let requirements = device.get_image_requirements(&image);

    let memory_type = get_memory_type_id::<B>(physical_device, requirements.type_mask, Properties::DEVICE_LOCAL)
        .expect("Could not find suitable memory type");

    let memory = device.allocate_memory(MemoryTypeId(memory_type), requirements.size)
        .expect("Could not allocate memory");

    device.bind_image_memory(&memory, 0, &mut image).expect("Could not bind image memory");

    return (image, memory);
}

pub unsafe fn create_image_view<B: gfx_hal::Backend>(device: &B::Device, image: &B::Image, info: ImageViewInfo) -> B::ImageView {
    let img_view = device.create_image_view(
        &image,
        info.kind,
        info.format,
        Swizzle::NO,
        SubresourceRange {
            aspects: info.aspects,
            level_start: 0,
            level_count: None,
            layer_start: 0,
            layer_count: None,
        }
    ).expect("Could not create image view");

    return img_view;
}

pub unsafe fn transition_image_layout<B: gfx_hal::Backend>(command_pool: &mut B::CommandPool, layout: Range<Layout>, image: &B::Image, queue: &mut B::CommandQueue) {
    let mut command_buffer = command_pool.allocate_one(Level::Primary);
    command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

    let (aspects, access_mask, stage) = if layout.start == Layout::Undefined && layout.end == Layout::TransferDstOptimal {
        (
            Aspects::COLOR,
            Access::empty()..Access::TRANSFER_WRITE,
            PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER
        )
    } else if layout.start == Layout::TransferDstOptimal && layout.end == Layout::ShaderReadOnlyOptimal {
        (
            Aspects::COLOR,
            Access::TRANSFER_WRITE..Access::TRANSFER_READ,
            PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER
        )
    } else if layout.start == Layout::Undefined && layout.end == Layout::DepthStencilAttachmentOptimal {
        (
            Aspects::DEPTH| Aspects::STENCIL,
            Access::empty()..(Access::DEPTH_STENCIL_ATTACHMENT_READ | Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
            PipelineStage::TOP_OF_PIPE..PipelineStage::EARLY_FRAGMENT_TESTS
        )
    } else {
        panic!("Could not transition image layout");
    };

    let barrier = Barrier::Image {
        states: (access_mask.start, layout.start)..(access_mask.end, layout.end),
        target: image,
        range: SubresourceRange {
            aspects,
            level_start: 0,
            level_count: None,
            layer_start: 0,
            layer_count: None,
        },
        families: None,
    };

    command_buffer.pipeline_barrier(stage, Dependencies::empty(), once(&barrier));
    command_buffer.finish();

    queue.submit_without_semaphores(once(&command_buffer), None);
    queue.wait_idle().expect("Could not wait on queue");

    command_pool.free(once(command_buffer));
}
