use gfx_hal::{adapter, buffer::Usage, device::Device, memory::Properties};

#[path = "helpers.rs"] mod helpers;

pub unsafe fn make_buffer<B: gfx_hal::Backend>(
    device: &B::Device,
    physical_device: &B::PhysicalDevice,
    buffer_len: usize,
    usage: Usage,
    properties: Properties,
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

pub fn make_buffer_from_vec<B: gfx_hal::Backend, T>(list: &Vec<T>, device: &B::Device, adapter: &adapter::Adapter<B>, usage: Usage) -> (B::Memory, B::Buffer) {
    let buffer_len = helpers::vec_size(&list);

    let (buffer_memory, buffer) = unsafe {
        make_buffer::<B>(
            device,
            &adapter.physical_device,
            buffer_len,
            usage,
            Properties::CPU_VISIBLE,
        )
    };

    unsafe {
        use gfx_hal::memory::Segment;

        let mapped_memory = device
            .map_memory(&buffer_memory, Segment::ALL)
            .expect("Failed to map memory");

        std::ptr::copy_nonoverlapping(list.as_ptr() as *const u8, mapped_memory, buffer_len);

        device
            .flush_mapped_memory_ranges(vec![(&buffer_memory, Segment::ALL)])
            .expect("Out of memory");

        device.unmap_memory(&buffer_memory);
    }

    return (buffer_memory, buffer);
}
