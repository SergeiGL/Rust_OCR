use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use std::path::Path;
use Rust_OCR::drawing::*;
use Rust_OCR::*;

const WIDTH: usize = 280; // Width of the window
const HEIGHT: usize = WIDTH; // Height of the window
const CANVAS_SIZE_PXLS: usize = 28; // Size of the canvas
const NEURONS_IN_HIDDEN_LAYER: usize = 10;
const PIXELS_PER_IMG: usize = CANVAS_SIZE_PXLS * CANVAS_SIZE_PXLS;


fn main() {
    let path_to_weights = format!("weights/neurons-{}_pixels-{}x{}.bin", NEURONS_IN_HIDDEN_LAYER, CANVAS_SIZE_PXLS, CANVAS_SIZE_PXLS);
    assert!(Path::new(&path_to_weights).exists(), "\nNo weights exist in the weights folder. Train the model first!\n");

    let weights = load_weights::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>(&path_to_weights);

    let mut last_mouse_pos: Option<(usize, usize)> = None;

    let zoom_scale_x = WIDTH / CANVAS_SIZE_PXLS;
    let zoom_scale_y = HEIGHT / CANVAS_SIZE_PXLS;
    let mut canvas: [u32; CANVAS_SIZE_PXLS * CANVAS_SIZE_PXLS] = [0xFFFFFF; CANVAS_SIZE_PXLS * CANVAS_SIZE_PXLS];

    let mut window = Window::new("Draw a Number", WIDTH, HEIGHT,
                                 WindowOptions {
                                     resize: false,
                                     ..WindowOptions::default()
                                 }).unwrap();

    loop {
        let zoomed_canvas = zoom_canvas::<PIXELS_PER_IMG>(&canvas, CANVAS_SIZE_PXLS, CANVAS_SIZE_PXLS, zoom_scale_x, zoom_scale_y, WIDTH, HEIGHT);

        if window.get_mouse_down(MouseButton::Left) {
            if let Some((x, y)) = window.get_mouse_pos(MouseMode::Discard) {
                let (x, y) = (x as usize, y as usize);
                if let Some((lx, ly)) = last_mouse_pos {
                    draw_line::<CANVAS_SIZE_PXLS, PIXELS_PER_IMG>(&mut canvas, lx / zoom_scale_x, ly / zoom_scale_y, x / zoom_scale_x, y / zoom_scale_y);
                }
                last_mouse_pos = Some((x, y));
            }
        } else {
            last_mouse_pos = None;
        }

        if !window.is_open() {
            break;
        }

        if window.is_key_pressed(Key::Enter, minifb::KeyRepeat::No) {
            let x_matrix = canvas_to_matrix::<PIXELS_PER_IMG>(&canvas);
            let params = forward_pass::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG, 1>(&x_matrix, &weights);
            println!("{:?}", get_predictions(&params.a3));
            clear_canvas(&mut canvas);
        }

        if window.is_key_pressed(Key::Backspace, minifb::KeyRepeat::No) {
            clear_canvas(&mut canvas);
        }

        window.update_with_buffer(&zoomed_canvas, WIDTH, HEIGHT).unwrap();
    }
}
