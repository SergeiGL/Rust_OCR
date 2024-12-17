use nalgebra::SMatrix;
use std::fs::File;
use std::io;
use std::io::prelude::*;

pub fn canvas_to_matrix<const PIXELS_PER_IMG: usize>(
    canvas: &[u32],
) -> SMatrix<f64, PIXELS_PER_IMG, 1>
{
    // Initialize a zero matrix with the specified dimensions
    let mut matrix = SMatrix::<f64, PIXELS_PER_IMG, 1>::zeros();

    // Iterate over the canvas and populate the matrix
    for (i, &value) in canvas.iter().enumerate() {
        matrix[i] = match value {
            0xFFFFFF => 0.0, // White pixel
            0x000000 => 1.0, // Black pixel
            _ => panic!("Invalid color value: {:#06X}", value),
        };
    }

    matrix
}


pub fn clear_canvas(canvas: &mut [u32]) {
    for el in canvas {
        *el = 0xFFFFFF;
    }
}


pub fn zoom_canvas<const PIXELS_PER_IMG: usize>(original: &[u32; PIXELS_PER_IMG], orig_width: usize, orig_height: usize, scale_x: usize, scale_y: usize, target_width: usize, target_height: usize) -> Vec<u32> {
    let mut zoomed = vec![0xFFFFFF; target_width * target_height]; // Default to white

    for orig_y in 0..orig_height {
        for orig_x in 0..orig_width {
            let color = original[orig_y * orig_width + orig_x];
            let start_x = orig_x * scale_x;
            let start_y = orig_y * scale_y;

            for dy in 0..scale_y {
                for dx in 0..scale_x {
                    let x = start_x + dx;
                    let y = start_y + dy;

                    if x < target_width && y < target_height {
                        zoomed[y * target_width + x] = color;
                    }
                }
            }
        }
    }
    zoomed
}


pub fn draw_line<const CANVAS_SIZE_PXLS: usize, const PIXELS_PER_IMG: usize>(canvas: &mut [u32; PIXELS_PER_IMG], x0: usize, y0: usize, x1: usize, y1: usize) {
    let dx = (x1 as isize - x0 as isize).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 as isize - y0 as isize).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x0 = x0 as isize;
    let mut y0 = y0 as isize;

    loop {
        if x0 >= 0 && x0 < CANVAS_SIZE_PXLS as isize && y0 >= 0 && y0 < CANVAS_SIZE_PXLS as isize {
            canvas[y0 as usize * CANVAS_SIZE_PXLS + x0 as usize] = 0x000000; // Draw black
        }

        if x0 == x1 as isize && y0 == y1 as isize {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}


pub fn save_canvas(canvas: &[u32], path: &str) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = io::BufWriter::new(file);

    let mut buffer = String::with_capacity(canvas.len());

    for &value in canvas {
        let mapped_value = match value {
            0xFFFFFF => '0',
            0x000000 => '1',
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unrecognized value in buffer")),
        };
        buffer.push(mapped_value);
    }

    writer.write_all(buffer.as_bytes())?;
    writer.flush()?;
    Ok(())
}
