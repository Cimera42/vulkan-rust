use std::fs;
use gfx_hal::{device::Device, pso::{EntryPoint, Specialization}};
use shaderc::ShaderKind;

fn compile_shader(glsl: &str, shader_kind: ShaderKind) -> Vec<u32> {
    let mut compiler = shaderc::Compiler::new().unwrap();

    let compiled_shader = compiler
        .compile_into_spirv(glsl, shader_kind, "unnamed", "main", None)
        .expect("Failed to compile shader");

    return compiled_shader.as_binary().to_vec();
}

pub unsafe fn create_entry<B: gfx_hal::Backend>(shader_module: &B::ShaderModule) -> EntryPoint<B> {
    return EntryPoint::<B> {
        entry: "main",
        module: &shader_module,
        specialization: Specialization::default(),
    };
}

pub unsafe fn create_module<B: gfx_hal::Backend>(device: &B::Device, filename: &str, kind: ShaderKind) -> B::ShaderModule {
    println!("{}", filename);
    let shader_ascii = fs::read_to_string(filename).expect("Could not load shader file");

    let shader_module = device
        .create_shader_module(&compile_shader(&shader_ascii, kind))
        .expect("Failed to create shader module");

    return shader_module;
}
