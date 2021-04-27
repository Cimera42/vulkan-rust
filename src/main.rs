// fn main() {
//     println!("Hello, world!");
// }

extern crate glfw;

use glfw::{Action, Context, Key};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
struct Window {
    name: String,
    width: u32,
    height: u32,
}

#[derive(Serialize, Deserialize)]
struct Config {
    window: Window,
}

fn load_config(filename: &str) -> Config {
    let config_string  = fs::read_to_string(filename).expect("Could not load config file");
    let config: Config = serde_json::from_str(&config_string).expect("JSON was not well-formatted");
    return config;
}

fn main() {
    let config = load_config("config.json");

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (mut window, events) = glfw.create_window(config.window.width, config.window.height, &config.window.name, glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.set_key_polling(true);
    window.make_current();

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}
