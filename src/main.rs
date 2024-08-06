use std::time::Instant;
use digit_ocr_nn::{*};



fn init_params(pixels_per_img: usize, neurons_in_hidden_layer: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>){
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let b_in: Vec<Vec<f32>> = (0..neurons_in_hidden_layer)
        .map(|_| vec![rng.gen_range(-0.5..=0.5)])
        .collect();

    let w_in: Vec<Vec<f32>> = (0..neurons_in_hidden_layer)
        .map(|_| (0..pixels_per_img)
            .map(|_| rng.gen_range(-0.5..=0.5))
            .collect())
        .collect();

    let b_h: Vec<Vec<f32>> = (0..neurons_in_hidden_layer)
        .map(|_| vec![rng.gen_range(-0.5..=0.5)])
        .collect();

    let w_h: Vec<Vec<f32>> = (0..neurons_in_hidden_layer)
        .map(|_| (0..neurons_in_hidden_layer)
            .map(|_| rng.gen_range(-0.5..=0.5))
            .collect())
        .collect();

    let b_out: Vec<Vec<f32>> = (0..10)
        .map(|_| vec![rng.gen_range(-0.5..=0.5)])
        .collect();

    let w_out: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..neurons_in_hidden_layer)
            .map(|_| rng.gen_range(-0.5..=0.5))
            .collect())
        .collect();

    (b_in, w_in, b_h, w_h, b_out, w_out)
}


fn backward_pass(
    x_ref: &Vec<Vec<f32>>,
    one_hot_ref: &Vec<Vec<f32>>,
    z1_ref: &Vec<Vec<f32>>,
    a1_ref: &Vec<Vec<f32>>,
    z2_ref: &Vec<Vec<f32>>,
    a2_ref: &Vec<Vec<f32>>,
    a3_ref: &Vec<Vec<f32>>,
    w_h_ref: &Vec<Vec<f32>>,
    w_out_ref: &Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {

    let one_over_m = 1_f32/(x_ref[0].len() as f32);

    let mut dz3 = a3_ref.clone();
    mtrix_minus_in_place(&mut dz3, one_hot_ref);
    mtrix_scale_by_c(2.0, &mut dz3);

    let mut dw_out = mtrix_dot(&dz3, &transpose(a2_ref));
    mtrix_scale_by_c(one_over_m, &mut dw_out);

    let mut db_out = mtrix_np_sum(&dz3);
    mtrix_scale_by_c(one_over_m, &mut db_out);

    let dz2 = mtrix_mult(&mtrix_dot(&transpose(w_out_ref), &dz3), &relu_deriv(&z2_ref));

    let mut dw_h = mtrix_dot(&dz2, &transpose(a1_ref));
    mtrix_scale_by_c(one_over_m, &mut dw_h);

    let mut db_h = mtrix_np_sum(&dz2);
    mtrix_scale_by_c(one_over_m, &mut db_h);

    let dz1 = mtrix_mult(&mtrix_dot(&transpose(w_h_ref), &dz2), &relu_deriv(&z1_ref));

    let mut dw_in = mtrix_dot(&dz1, &transpose(x_ref));
    mtrix_scale_by_c(one_over_m, &mut dw_in);

    let mut db_in = mtrix_np_sum(&dz1);
    mtrix_scale_by_c(one_over_m, &mut db_in);

    (db_in, dw_in, db_h, dw_h, db_out, dw_out)
}


fn update_params(b_in: &mut Vec<Vec<f32>>, db_in: &mut Vec<Vec<f32>> , w_in: &mut Vec<Vec<f32>>, dw_in: &mut Vec<Vec<f32>>,
                 b_h: &mut Vec<Vec<f32>>, db_h: &mut Vec<Vec<f32>>, w_h: &mut Vec<Vec<f32>>, dw_h: &mut Vec<Vec<f32>>,
                 b_out: &mut Vec<Vec<f32>>, db_out: &mut Vec<Vec<f32>>, w_out: &mut Vec<Vec<f32>>, dw_out: &mut Vec<Vec<f32>>,
                 lr: f32){

    mtrix_scale_by_c(lr, db_in);
    mtrix_scale_by_c(lr, dw_in);
    mtrix_scale_by_c(lr, db_h);
    mtrix_scale_by_c(lr, dw_h);
    mtrix_scale_by_c(lr, db_out);
    mtrix_scale_by_c(lr, dw_out);

    mtrix_minus_in_place(b_in, db_in);
    mtrix_minus_in_place(w_in, dw_in);
    mtrix_minus_in_place(b_h, db_h);
    mtrix_minus_in_place(w_h, dw_h);
    mtrix_minus_in_place(b_out, db_out);
    mtrix_minus_in_place(w_out, dw_out);
}


fn main(){
    let conf: Config = get_config();

    let pixels_per_img: usize = conf.image.canvas_size_pxls.pow(2);

    let path_to_data = "data/";
    assert_eq!(is_path_exist(path_to_data), true);

    let mut path_to_weights = String::from("weights/");
    assert_eq!(is_path_exist(&path_to_weights), true);
    path_to_weights.push_str(&format!("neurons-{}_pixels-{}x{}.bin", conf.neural_net.neurons_in_hidden_layer, conf.image.canvas_size_pxls, conf.image.canvas_size_pxls));

    let (x_matrix, y_matrix) = read_file_to_matrix(path_to_data, count_total_number_of_files(), pixels_per_img).unwrap();
    // println!("X matrix shape is: {}, {}", x_matrix.len(), x_matrix[0].len());
    let one_hot_y = one_hot(&y_matrix);

    let (mut b_in, mut w_in, mut b_h, mut w_h, mut b_out, mut w_out) =
        match is_path_exist(&path_to_weights) {
            true => {
                println!("Loading weights...");
                load_weights(&path_to_weights).unwrap()
            },
            false => {
                println!("Initializing weights...");
                init_params(pixels_per_img, conf.neural_net.neurons_in_hidden_layer)
            }
        };
    println!("Weights are set!");

    for epoch_n in 1..(conf.neural_net.total_epochs+1){
        let start = Instant::now();
        let (z1, a1, z2, a2, a3) =
            forward_pass(&b_in, &w_in, &b_h, &w_h, &b_out, &w_out, &x_matrix);

        let (mut db_in, mut dw_in, mut db_h, mut dw_h, mut db_out, mut dw_out) =
            backward_pass(&x_matrix, &one_hot_y, &z1, &a1, &z2, &a2, &a3, &w_h, &w_out);

        update_params(&mut b_in, &mut db_in,
                      &mut w_in, &mut dw_in,
                      &mut b_h, &mut db_h,
                      &mut w_h, &mut dw_h,
                      &mut b_out, &mut db_out,
                      &mut w_out, &mut dw_out,
                      conf.neural_net.lr);

        if epoch_n % 10 == 0{
            save_weights(&b_in, &w_in, &b_h, &w_h, &b_out, &w_out, &path_to_weights).expect("Unable to save weights");
            println!("Weights saved!");
        }

        println!("Accuracy: {}", get_accuracy(&get_predictions(&a3), &y_matrix));
        println!("Epoch: {} : {:?}", epoch_n, start.elapsed());
    }
}
