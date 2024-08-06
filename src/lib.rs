use std::path::Path;
use std::fs::{File, read_dir, create_dir};
use std::io;
use std::io::{BufReader, BufWriter, Read, Write};
use rayon::prelude::*;
use byteorder::{LittleEndian, ReadBytesExt};
use serde_derive::Deserialize;
use config::Config as ConfigBuilder;
use lazy_static::lazy_static;
use std::sync::RwLock;

#[cfg(test)]
mod tests;

const CANVAS_SIZE: usize = 28;


#[derive(Debug, Deserialize, Clone)]
pub struct ImageConfig {
    pub canvas_size_pxls: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NeuralNetConfig {
    pub neurons_in_hidden_layer: usize,
    pub total_epochs: usize,
    pub lr: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub image: ImageConfig,
    pub neural_net: NeuralNetConfig,
}

lazy_static!{
    pub static ref SETTINGS: RwLock<Config> = RwLock::new(load_config());
}

pub fn load_config() -> Config {
    let config = ConfigBuilder::builder()
        .add_source(config::File::with_name("config"))
        .build()
        .expect("Failed to build configuration");

    config.try_deserialize().expect("Failed to deserialize configuration")
}

pub fn get_config() -> Config {
    SETTINGS.read().unwrap().clone()
}

pub fn is_path_exist(path: &str) -> bool{
    let path = Path::new(path);
    path.exists()
}

pub fn create_fldrs_if_not_exist(base_dir: &str, flders_to_create: Vec<String>) {
    for i in flders_to_create{
        let folder_name = format!("{}{}", base_dir, i);
        let path = Path::new(&folder_name);

        if !path.exists() {
            match create_dir(path) {
                Ok(_) => println!("Created directory: {}", folder_name),
                Err(e) => println!("Failed to create directory {}: {}", folder_name, e),
            }
        } else {
            println!("Directory {} already exists", folder_name);
        }
    }
}

pub fn count_total_number_of_files() -> usize{
    fn count_files_in_directory(path: &str) -> Result<usize, std::io::Error> {
        let path = Path::new(path);
        let mut file_count = 0;

        if path.is_dir() {
            for entry in read_dir(path)? {
                let entry = entry?;
                let file_type = entry.file_type()?;

                if file_type.is_file() {
                    file_count += 1;
                }
            }
        } else {
            println!("The provided path is not a directory.");
        }

        Ok(file_count)
    }

    let mut result = 0;

    for i in 0..10{
        result+=count_files_in_directory(&format!("data/{i}/")).unwrap();
    }

    result
}


pub fn one_hot(y: &Vec<usize>) -> Vec<Vec<f32>> {
    let num_classes = 10;
    let len = y.len();

    let mut one_hot_y: Vec<Vec<f32>> = vec![vec![0.0; len]; num_classes];

    one_hot_y.par_iter_mut().enumerate().for_each(|(label, row)| {
        y.iter()
            .enumerate()
            .filter(|&(_, &y_label)| y_label == label)
            .for_each(|(i, _)| {
                row[i] = 1.0_f32;
            });
    });

    one_hot_y
}


pub fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut transposed = vec![vec![0.0; rows]; cols];

    transposed.par_iter_mut().enumerate().for_each(|(j, row)| {
        row.iter_mut().enumerate().for_each(|(i, elem)| {
            *elem = matrix[i][j];
        });
    });

    transposed
}


pub fn mtrix_minus_in_place(left: &mut [Vec<f32>], right: &[Vec<f32>]) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left[0].len(), right[0].len());

    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left_row, right_row)| {
            left_row.iter_mut()
                .zip(right_row.iter())
                .for_each(|(left_elem, right_elem)| {
                    *left_elem -= right_elem;
                });
        });
}


pub fn mtrix_mult(left: &Vec<Vec<f32>>, right: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(left.len(), right.len());
    assert_eq!(left[0].len(), right[0].len());

    let result: Vec<Vec<f32>> = left
        .par_iter()
        .zip(right.par_iter())
        .map(|(row_left, row_right)| {
            row_left
                .par_iter()
                .zip(row_right.par_iter())
                .map(|(val_left, val_right)| val_left * val_right)
                .collect()
        })
        .collect();

    result
}


pub fn mtrix_dot(left: &Vec<Vec<f32>>, right: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(left[0].len(), right.len());

    let left_len = left.len();
    let right_inner_len = right.len();
    let right_len = right[0].len();

    let mut result = vec![vec![0.0_f32; right_len]; left_len];

    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..right_len {
            for k in 0..right_inner_len {
                row[j] += left[i][k] * right[k][j];
            }
        }
    });

    result
}

pub fn mtrix_scale_by_c(cnst: f32, matrix: &mut Vec<Vec<f32>>) {
    matrix.par_iter_mut().for_each(|row| {
        row.par_iter_mut().for_each(|elem| {
            *elem *= cnst;
        });
    });
}

pub fn mtrix_np_sum(mtrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    mtrix.par_iter()
        .map(|row| vec![row.par_iter().sum::<f32>()])
        .collect()
}

pub fn relu_deriv(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    matrix.par_iter()
        .map(|row| {
            row.par_iter()
                .map(|&val| (val > 0.0) as u8 as f32)
                .collect()
        })
        .collect()
}


pub fn linear_combination(b: &Vec<Vec<f32>>, w: &Vec<Vec<f32>>, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
    assert_eq!(w[0].len(), x.len());
    assert_eq!(w.len(), b.len());

    let m = w.len();
    let n = x[0].len();

    let mut result = vec![vec![0.0; n]; m];

    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..n {
            row[j] = w[i].iter().zip(x.iter()).map(|(wi_k, x_kj)| wi_k * x_kj[j]).sum::<f32>() + b[i][0];
        }
    });
    result
}


pub fn relu(x: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
    x.par_iter()
        .map(|row| {
            row.par_iter()
                .map(|&elem| elem.max(0.0))
                .collect()
        })
        .collect()
}



// np.exp(Z) / sum(np.exp(Z))
pub fn softmax(x: &mut Vec<Vec<f32>>) {
    let num_cols = x[0].len();

    let max_values: Vec<f32> = (0..num_cols)
        .into_par_iter()
        .map(|col| x.iter().map(|row| row[col]).fold(f32::NEG_INFINITY, f32::max))
        .collect();

    x.par_iter_mut().for_each(|row| {
        row.iter_mut().enumerate().for_each(|(col, val)| {
            *val = f32::exp(*val - max_values[col]);
        });
    });

    let column_sums: Vec<f32> = (0..num_cols)
        .into_par_iter()
        .map(|col| x.iter().map(|row| row[col]).sum())
        .collect();

    x.par_iter_mut().for_each(|row| {
        row.iter_mut().enumerate().for_each(|(col, val)| {
            *val /= column_sums[col];
        });
    });
}


pub fn get_predictions(a2: &Vec<Vec<f32>>) -> Vec<usize> {
    (0..a2[0].len()).into_par_iter().map(|col| {
        let mut prediction: usize = 0;
        let mut max_val = a2[0][col];
        for row in 1..a2.len() {
            if a2[row][col] > max_val {
                max_val = a2[row][col];
                prediction = row;
            }
        }
        prediction
    }).collect()
}


pub fn get_accuracy(predictions: &Vec<usize>, y_matrix: &Vec<usize>) -> f32{
    assert_eq!(predictions.len(), y_matrix.len());

    let sum: usize = predictions.par_iter()
                                .zip(y_matrix.par_iter())
                                .filter(|(p, y)| p == y)
                                .count();

    (sum as f32) / (predictions.len() as f32)
}


pub fn read_file_to_matrix(path: &str, n_file: usize, pixels_per_img: usize) -> Result<(Vec<Vec<f32>>, Vec<usize>), Box<dyn std::error::Error>> {
    println!("Total number of files is: {n_file}");

    // 2d array with vectors(slow af) of u8, all values are initialized 0
    let mut picture_n = 0;
    let mut x_matrix = vec![vec![0_f32; pixels_per_img]; n_file];
    let mut y_matrix = vec![0_usize; n_file];

    for entry in read_dir(path)? {
        let entry = entry?;
        let sub_path = entry.path();

        if sub_path.is_dir() {
            for sub_entry in read_dir(&sub_path)? {
                let correct_answer: f32 = String::from(
                    sub_path.to_str()
                    .unwrap()
                    .split('/')
                    .last()
                    .unwrap()
                ).parse().unwrap();

                let sub_entry = sub_entry?;
                let img_path = format!("{}", sub_entry.path().display());

                let file = File::open(img_path)?;
                let mut reader = BufReader::new(file);
                let mut ln = String::new();

                // read file to EOF
                let a = reader.read_to_string(&mut ln)?;
                if picture_n ==0{
                    println!("String read with length: {}", a);
                }

                // remove newlines, this call modifies String directly
                ln.retain(|z| z != '\n');

                //enumerate starts with 0
                for (i, number) in ln.chars().enumerate() {
                    match number{
                        '0' => continue,
                        '1' => x_matrix[picture_n][i]=1.0_f32,
                        _ => panic!()
                    }
                }
                y_matrix[picture_n]=correct_answer as usize;
                picture_n+=1;
            }
        }
    }
    Ok((transpose(&x_matrix), y_matrix))
}



pub fn forward_pass(b_in: &Vec<Vec<f32>>, w_in: &Vec<Vec<f32>>, b_h: &Vec<Vec<f32>>, w_h: &Vec<Vec<f32>>, b_out: &Vec<Vec<f32>>, w_out: &Vec<Vec<f32>>, x_matrix: &Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)  {
    let z1 = linear_combination(b_in, w_in, x_matrix);
    let a1 = relu(&z1);

    let z2 = linear_combination(b_h, w_h, &a1);
    let a2 = relu(&z2);

    let mut a3 = linear_combination(b_out, w_out, &a2);
    softmax(&mut a3);
    (z1, a1, z2, a2, a3)
}


pub fn save_weights(b_in: &[Vec<f32>], w_in: &[Vec<f32>], b_h: &[Vec<f32>], w_h: &[Vec<f32>], b_out: &[Vec<f32>], w_out: &[Vec<f32>], path: &str) -> io::Result<()> {
    let path = Path::new(path);
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    fn write_f32_slice(writer: &mut dyn Write, slice: &[f32]) -> io::Result<()> {
        for &value in slice {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    let dimensions = [
        b_in.len() as u32,
        b_in[0].len() as u32,
        w_in.len() as u32,
        w_in[0].len() as u32,
        b_h.len() as u32,
        b_h[0].len() as u32,
        w_h.len() as u32,
        w_h[0].len() as u32,
        b_out.len() as u32,
        b_out[0].len() as u32,
        w_out.len() as u32,
        w_out[0].len() as u32,
    ];

    for &dim in &dimensions {
        writer.write_all(&dim.to_le_bytes())?;
    }

    for matrix in &[b_in, w_in, b_h, w_h, b_out, w_out] {
        for row in *matrix {
            write_f32_slice(&mut writer, row)?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub fn load_weights(path: &str) -> io::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    fn read_matrix(reader: &mut impl Read, matrix: &mut Vec<Vec<f32>>) -> io::Result<()> {
        for row in matrix.iter_mut() {
            for value in row.iter_mut() {
                *value = reader.read_f32::<LittleEndian>()?;
            }
        }
        Ok(())
    }

    let file = File::open(Path::new(path))?;
    let mut reader = BufReader::new(file);

    let dimensions: Vec<usize> = (0..12)
        .map(|_| reader.read_u32::<LittleEndian>().map(|d| d as usize))
        .collect::<io::Result<_>>()?;

    let mut b_in = vec![vec![0.0; dimensions[1]]; dimensions[0]];
    let mut w_in = vec![vec![0.0; dimensions[3]]; dimensions[2]];
    let mut b_h = vec![vec![0.0; dimensions[5]]; dimensions[4]];
    let mut w_h = vec![vec![0.0; dimensions[7]]; dimensions[6]];
    let mut b_out = vec![vec![0.0; dimensions[9]]; dimensions[8]];
    let mut w_out = vec![vec![0.0; dimensions[11]]; dimensions[10]];

    read_matrix(&mut reader, &mut b_in)?;
    read_matrix(&mut reader, &mut w_in)?;

    read_matrix(&mut reader, &mut b_h)?;
    read_matrix(&mut reader, &mut w_h)?;

    read_matrix(&mut reader, &mut b_out)?;
    read_matrix(&mut reader, &mut w_out)?;
    Ok((b_in, w_in, b_h, w_h, b_out, w_out))
}


pub fn save_canvas(canvas: &[u32], path: &str) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

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


pub fn canvas_to_matrix(canvas: &[u32]) -> Vec<Vec<f32>>{
    let chunk_count = (canvas.len() + CANVAS_SIZE - 1) / CANVAS_SIZE;
    let mut result = Vec::with_capacity(chunk_count * CANVAS_SIZE);

    let white = vec![0.0];
    let black = vec![1.0];

    for &value in canvas {
        match value {
            0xFFFFFF => result.push(white.clone()),
            0x000000 => result.push(black.clone()),
            _ => panic!(),
        }
    }

    result
}


pub fn clear_canvas(canvas: &mut [u32]) {
    for el in canvas{
        *el = 0xFFFFFF;
    }
}


// gpt shit
pub fn zoom_canvas(original: &[u32; CANVAS_SIZE * CANVAS_SIZE], orig_width: usize, orig_height: usize, scale_x: usize, scale_y: usize, target_width: usize, target_height: usize) -> Vec<u32> {
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


pub fn draw_line(canvas: &mut [u32; CANVAS_SIZE * CANVAS_SIZE], x0: usize, y0: usize, x1: usize, y1: usize) {
    let dx = (x1 as isize - x0 as isize).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 as isize - y0 as isize).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x0 = x0 as isize;
    let mut y0 = y0 as isize;

    loop {
        if x0 >= 0 && x0 < CANVAS_SIZE as isize && y0 >= 0 && y0 < CANVAS_SIZE as isize {
            canvas[y0 as usize * CANVAS_SIZE + x0 as usize] = 0x000000; // Draw black
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
