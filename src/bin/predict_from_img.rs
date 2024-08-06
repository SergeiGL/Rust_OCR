use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use digit_ocr_nn::{*};

const WIDTH: usize = 280; // Width of the window
const HEIGHT: usize = WIDTH; // Height of the window
const CANVAS_SIZE_PXLS: usize = 28; // Size of the canvas


fn main(){
    let conf: Config = get_config();
    let path_to_weights = format!("weights/neurons-{}_pixels-{}x{}.bin", conf.neural_net.neurons_in_hidden_layer, CANVAS_SIZE_PXLS, CANVAS_SIZE_PXLS);
    assert_eq!(is_path_exist(&path_to_weights), true);

    let (b_in, w_in, b_h, w_h, b_out, w_out) = load_weights(&path_to_weights).unwrap();

    let mut last_mouse_pos: Option<(usize, usize)> = None;

    // Zoom the 30x30 canvas to fit the window size
    let zoom_scale_x = WIDTH / CANVAS_SIZE_PXLS;
    let zoom_scale_y = HEIGHT / CANVAS_SIZE_PXLS;
    let mut canvas: [u32; CANVAS_SIZE_PXLS * CANVAS_SIZE_PXLS] = [0xFFFFFF; CANVAS_SIZE_PXLS * CANVAS_SIZE_PXLS];

    let mut window = Window::new("Draw a Number",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        }).unwrap();

    loop {
        let zoomed_canvas = zoom_canvas(&canvas, CANVAS_SIZE_PXLS, CANVAS_SIZE_PXLS, zoom_scale_x, zoom_scale_y, WIDTH, HEIGHT);

        if window.get_mouse_down(MouseButton::Left){
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
            break;
        }

        if window.is_key_pressed(Key::Enter, minifb::KeyRepeat::No) {
            let x_matrix = canvas_to_matrix(&canvas);
            let (_z1, _a1, _z2, _a2, a3) = forward_pass(&b_in, &w_in, &b_h, &w_h, &b_out, &w_out, &x_matrix);
            println!("{:?}", get_predictions(&a3));
            clear_canvas(&mut canvas);
        }

        if window.is_key_pressed(Key::Backspace, minifb::KeyRepeat::No) {
            clear_canvas(&mut canvas);
        }

        window.update_with_buffer(&zoomed_canvas, WIDTH, HEIGHT).unwrap();
    }
}
