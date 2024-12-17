use nalgebra::{SMatrix, SVector};
use std::fs::{create_dir, read_dir, File};
use std::io;
use std::io::Read;
use std::path::Path;

pub fn is_path_exist(path: &str) -> bool {
    let path = Path::new(path);
    path.exists()
}

pub fn create_fldrs_if_not_exist(base_dir: &str, flders_to_create: Vec<String>) {
    for i in flders_to_create {
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

pub fn count_total_number_of_files() -> usize {
    fn count_files_in_directory(path: &str) -> Result<usize, io::Error> {
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
            println!("The provided path is not a directory: {path:?}");
        }

        Ok(file_count)
    }

    let mut result = 0;

    for i in 0..10 { // as there are 0..9 digits to recognise
        result += count_files_in_directory(&format!("data/{i}/")).expect("");
    }

    result
}


pub fn one_hot<const N_PICTURES: usize>(y: &SVector<u8, N_PICTURES>) -> SMatrix<f64, 10, N_PICTURES> {
    let mut one_hot_y = SMatrix::<f64, 10, N_PICTURES>::repeat(0.0);

    for (exp_id, &true_label) in y.iter().enumerate() {
        one_hot_y[(true_label as usize, exp_id)] = 1.0;
    }

    one_hot_y
}


pub fn read_file_to_matrix<const PIXELS_PER_IMG: usize, const N_PICTURES: usize>(
    path: &str
) -> Result<(
    SMatrix<f64, PIXELS_PER_IMG, N_PICTURES>,   // x_matrix
    SVector<u8, N_PICTURES>                     // y_matrix
),
    Box<dyn std::error::Error>>
{
    println!("Total number of files is: {N_PICTURES}");

    let mut picture_n: usize = 0;
    let mut x_matrix = SMatrix::<f64, PIXELS_PER_IMG, N_PICTURES>::zeros();
    let mut y_matrix = SVector::<u8, N_PICTURES>::zeros();

    for entry in read_dir(path)? {
        let entry = entry?;
        let sub_path = entry.path();

        if sub_path.is_dir() {
            for sub_entry in read_dir(&sub_path)? {
                let correct_answer: u8 = String::from(
                    sub_path.to_str()
                        .unwrap()
                        .split('/')
                        .last()
                        .unwrap()
                ).parse().unwrap();

                let sub_entry = sub_entry?;
                let img_path = format!("{}", sub_entry.path().display());

                let file = File::open(img_path)?;
                let mut reader = io::BufReader::new(file);
                let mut ln = String::new();

                // read file to EOF
                let a = reader.read_to_string(&mut ln)?;
                if picture_n == 0 {
                    println!("String read with length: {}", a);
                }

                // remove newlines, this call modifies String directly
                ln.retain(|z| z != '\n');

                // enumerate starts with 0
                for (pixl_i, number) in ln.chars().enumerate() {
                    x_matrix[(pixl_i, picture_n)] = (number as u8 - '0' as u8) as f64;
                }
                y_matrix[picture_n] = correct_answer;
                picture_n += 1;
            }
        }
    }
    Ok((x_matrix, y_matrix))
}
