use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use digit_ocr_nn::{*};


const WIDTH: usize = 280; // Width of the window
const HEIGHT: usize = WIDTH; // Height of the window
const CANVAS_SIZE: usize = 28; // Size of the canvas, 30x30



fn main() {
    create_fldrs_if_not_exist("data/", (0..10).map(|i| i.to_string()).collect());
    let mut number_to_draw:i32 = 0;

    let zoom_scale_x = WIDTH / CANVAS_SIZE;
    let zoom_scale_y = HEIGHT / CANVAS_SIZE;

    println!("Total numer of training examples: {}", count_total_number_of_files());

    'main: loop {
        let mut last_mouse_pos: Option<(usize, usize)> = None;
        let mut canvas: [u32; CANVAS_SIZE * CANVAS_SIZE] = [0xFFFFFF; CANVAS_SIZE * CANVAS_SIZE];

        let mut window = Window::new(&format!("{}", number_to_draw),
            WIDTH,
            HEIGHT,
            WindowOptions {
                resize: false,
                ..WindowOptions::default()
            }).unwrap();

        'number_drawing: loop {
            let zoomed_canvas = zoom_canvas(&canvas, CANVAS_SIZE, CANVAS_SIZE, zoom_scale_x, zoom_scale_y, WIDTH, HEIGHT);

            if window.get_mouse_down(MouseButton::Left) {
                if let Some((x, y)) = window.get_mouse_pos(MouseMode::Discard) {
                    let (x, y) = (x as usize, y as usize);
                    if let Some((lx, ly)) = last_mouse_pos {
                        draw_line(&mut canvas, lx / zoom_scale_x, ly / zoom_scale_y, x / zoom_scale_x, y / zoom_scale_y);
                    }
                    last_mouse_pos = Some((x, y));
                }
            } else {
                last_mouse_pos = None;
            }

            if !window.is_open() {
                break 'main;
            }

            if window.is_key_pressed(Key::Enter, minifb::KeyRepeat::No) {
                save_canvas(
                    &canvas,
                    &format!("data/{}/{}",
                    number_to_draw,
                    count_total_number_of_files() + 1))
                .unwrap();
                clear_canvas(&mut canvas);
            }

            if window.is_key_pressed(Key::Backspace, minifb::KeyRepeat::No) {
                clear_canvas(&mut canvas);
            }

            if window.is_key_pressed(Key::Right, minifb::KeyRepeat::No) {
                if number_to_draw >= 9 { break 'main; }
                else {
                    number_to_draw += 1;
                    break 'number_drawing;
                }
            }

            if window.is_key_pressed(Key::Left, minifb::KeyRepeat::No) {
                if number_to_draw <= 0 { break 'main; }
                else {
                    number_to_draw -= 1;
                    break 'number_drawing;
                }
            }
            window.update_with_buffer(&zoomed_canvas, WIDTH, HEIGHT).unwrap();
        }
    }
}
