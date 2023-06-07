use gfx_hal::{adapter, buffer::Usage, device::Device};

#[path = "helpers.rs"] mod helpers;
#[path = "gfx_helpers.rs"] mod gfx_helpers;

#[repr(C)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

pub struct Mesh<B: gfx_hal::Backend> {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub vertex_buffer_memory: B::Memory,
    pub vertex_buffer: B::Buffer,

    pub index_buffer_memory: B::Memory,
    pub index_buffer: B::Buffer,
}

impl<B: gfx_hal::Backend> Mesh<B> {
    pub fn new(filename: &str, device: &B::Device, adapter: &adapter::Adapter<B>) -> Result<Self, ()> {
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

                let (vertex_buffer_memory, vertex_buffer) = gfx_helpers::make_buffer_from_vec(&vertices, device, adapter, Usage::VERTEX);
                let (index_buffer_memory, index_buffer) = gfx_helpers::make_buffer_from_vec(&indices, device, adapter, Usage::INDEX);

                return Ok(Mesh {
                    vertices,
                    indices,
                    vertex_buffer_memory, vertex_buffer,
                    index_buffer_memory, index_buffer,
                })
            }
        }
        return Err(());
    }

    pub unsafe fn free(mesh: Self, device: &B::Device) {
        device.free_memory(mesh.vertex_buffer_memory);
        device.destroy_buffer(mesh.vertex_buffer);

        device.free_memory(mesh.index_buffer_memory);
        device.destroy_buffer(mesh.index_buffer);
    }
}
