use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io;

pub mod drawing;
pub mod utils;


#[derive(Serialize, Deserialize, Debug)]
pub struct WeightsStruct<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize> {
    b_in: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, 1>,
    w_in: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
    b_h: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, 1>,
    w_h: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, NEURONS_IN_HIDDEN_LAYER>,
    b_out: SMatrix<f64, 10, 1>,
    w_out: SMatrix<f64, 10, NEURONS_IN_HIDDEN_LAYER>,
}


pub struct DWeightsStruct<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize> {
    db_in: f64,
    dw_in: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
    db_h: f64,
    dw_h: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, NEURONS_IN_HIDDEN_LAYER>,
    db_out: f64,
    dw_out: SMatrix<f64, 10, NEURONS_IN_HIDDEN_LAYER>,
}


pub struct ParamsStruct<const NEURONS_IN_HIDDEN_LAYER: usize, const N_PICTURES: usize> {
    z1: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, N_PICTURES>,
    a1: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, N_PICTURES>,
    z2: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, N_PICTURES>,
    a2: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, N_PICTURES>,
    pub a3: SMatrix<f64, 10, N_PICTURES>,
}


pub fn init_weights<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize>()
    -> WeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>
{
    let b_in = SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, 1>::new_random() - SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, 1>::from_element(0.5);
    let w_in = SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>::new_random() - SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>::from_element(0.5);

    let b_h = SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, 1>::new_random() - SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, 1>::from_element(0.5);
    let w_h = SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, NEURONS_IN_HIDDEN_LAYER>::new_random() - SMatrix::<f64, NEURONS_IN_HIDDEN_LAYER, NEURONS_IN_HIDDEN_LAYER>::from_element(0.5);

    let b_out = SMatrix::<f64, 10, 1>::new_random() - SMatrix::<f64, 10, 1>::from_element(0.5);
    let w_out = SMatrix::<f64, 10, NEURONS_IN_HIDDEN_LAYER>::new_random() - SMatrix::<f64, 10, NEURONS_IN_HIDDEN_LAYER>::from_element(0.5);

    WeightsStruct::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG> { b_in, w_in, b_h, w_h, b_out, w_out }
}


pub fn load_weights<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize>(
    path: &str,
) -> WeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG> {
    // Open the file in read mode
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);

    // Deserialize the data from the file using bincode
    bincode::deserialize_from(reader).unwrap()
}

pub fn save_weights<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize>(
    weights: &WeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
    path: &str,
) -> io::Result<()> {
    // Open the file in write mode
    let file = File::create(path)?;
    let writer = io::BufWriter::new(file);

    // Serialize the data into the file using bincode
    bincode::serialize_into(writer, weights)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

pub fn forward_pass<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize, const N_PICTURES: usize>(
    x_matrix: &SMatrix<f64, PIXELS_PER_IMG, N_PICTURES>,
    weights: &WeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
) ->
    ParamsStruct<NEURONS_IN_HIDDEN_LAYER, N_PICTURES>
{
    // np.exp(Z) / sum(np.exp(Z))
    pub fn softmax<const N_PICTURES: usize>(matrix: &mut SMatrix::<f64, 10, N_PICTURES>) {
        matrix.iter_mut().for_each(|x| *x = x.exp());

        let col_sum = matrix.row_sum();

        for mut row in matrix.row_iter_mut() {
            row.component_div_assign(&col_sum);
        }
    }

    let b_in_matrix: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, N_PICTURES> = SMatrix::from_fn(|row, _| weights.b_in[row]);
    let z1 = weights.w_in * x_matrix + b_in_matrix;
    let a1 = z1.map(|elem| elem.max(0.0));

    let b_h_matrix: SMatrix<f64, NEURONS_IN_HIDDEN_LAYER, N_PICTURES> = SMatrix::from_fn(|row, _| weights.b_h[row]);
    let z2 = weights.w_h * a1 + b_h_matrix;
    let a2 = z2.map(|elem| elem.max(0.0));

    let b_out_matrix: SMatrix<f64, 10, N_PICTURES> = SMatrix::from_fn(|row, _| weights.b_out[row]);
    let mut a3 = weights.w_out * a2 + b_out_matrix;
    softmax(&mut a3);

    ParamsStruct::<NEURONS_IN_HIDDEN_LAYER, N_PICTURES> { z1, a1, z2, a2, a3 }
}


pub fn backward_pass<const PIXELS_PER_IMG: usize, const N_PICTURES: usize, const NEURONS_IN_HIDDEN_LAYER: usize>(
    x: &SMatrix<f64, PIXELS_PER_IMG, N_PICTURES>,
    one_hot_y: &SMatrix<f64, 10, N_PICTURES>,
    weights: &WeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
    params: &ParamsStruct<NEURONS_IN_HIDDEN_LAYER, N_PICTURES>,
) ->
    DWeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>
{
    let one_over_m = 1_f64 / N_PICTURES as f64;

    let dz3 = (params.a3 - one_hot_y).scale(2.0);

    let dw_out = (dz3 * params.a2.transpose()).scale(one_over_m);

    let db_out = dz3.sum() * one_over_m;

    let dz2 = (weights.w_out.transpose() * dz3).component_mul(&params.z2.map(|z| if z > 0.0 { 1.0 } else { 0.0 }));

    let dw_h = (dz2 * params.a1.transpose()).scale(one_over_m);

    let db_h = dz2.sum() * one_over_m;

    let dz1 = (weights.w_h.transpose() * dz2).component_mul(&params.z1.map(|z| if z > 0.0 { 1.0 } else { 0.0 }));

    let dw_in = (dz1 * x.transpose()).scale(one_over_m);

    let db_in = dz1.sum() * one_over_m;

    DWeightsStruct::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG> {
        db_in,
        dw_in,
        db_h,
        dw_h,
        db_out,
        dw_out,
    }
}


pub fn update_params<const NEURONS_IN_HIDDEN_LAYER: usize, const PIXELS_PER_IMG: usize>(
    weigh: &mut WeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
    d: &DWeightsStruct<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>,
    lr: f64,
) {
    weigh.b_in.apply(|b| *b -= d.db_in * lr);
    weigh.w_in -= d.dw_in.scale(lr);

    weigh.b_h.apply(|b| *b -= d.db_h * lr);
    weigh.w_h -= d.dw_h.scale(lr);

    weigh.b_out.apply(|b| *b -= d.db_out * lr);
    weigh.w_out -= d.dw_out.scale(lr);
}

pub fn get_predictions<const N_PICTURES: usize>(
    a3: &SMatrix<f64, 10, N_PICTURES>
) ->
    SVector<u8, N_PICTURES>
{
    SVector::<u8, N_PICTURES>::from_fn(|col, _| a3.column(col).argmax().0 as u8)
}


pub fn get_accuracy<const N_PICTURES: usize>(
    predictions: &SVector<u8, N_PICTURES>,
    y_matrix: &SVector<u8, N_PICTURES>,
) -> f64 {
    // println!("predictions: {:?}", predictions.sum());
    let correct = predictions
        .iter()
        .zip(y_matrix.iter())
        .filter(|(&p, &y)| p == y)
        .count();

    correct as f64 / N_PICTURES as f64
}