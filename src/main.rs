use Rust_OCR::*;
pub mod utils;
use utils::*;


const CANVAS_SIZE_PXLS: usize = 28;
const NEURONS_IN_HIDDEN_LAYER: usize = 10;
const TOTAL_EPOCHS: usize = 5_000;
const LR: f64 = 0.04;
const PIXELS_PER_IMG: usize = CANVAS_SIZE_PXLS * CANVAS_SIZE_PXLS;
const N_PICTURES: usize = 28_000;
const PATH_TO_DATA: &str = "data/";

fn main() {
    let builder = std::thread::Builder::new()
        .name("increased_stack".into())
        .stack_size(5000 * 1024 * 1024); // 5 GB of stack space

    let handler = builder.spawn(|| {
        assert_eq!(is_path_exist(PATH_TO_DATA), true, "No data folder exists. Create it by following these steps:\n
                                                    1) Move to the useful_resources folder
                                                    2) Unzip the train.zip to the same folder (useful_resources)
                                                    3) Run the first section of the the .ipynb file
                                                    4) Move the resulting data folder 1 layer up so it is next to the useful_resources and weights folders."
        );

        let mut path_to_weights = String::from("weights/");
        assert_eq!(is_path_exist(&path_to_weights), true, "Create folder weights next to the useful_resources and data folders.");
        path_to_weights.push_str(&format!("neurons-{}_pixels-{}x{}.bin", NEURONS_IN_HIDDEN_LAYER, CANVAS_SIZE_PXLS, CANVAS_SIZE_PXLS));

        let (x_matrix, y_matrix) = read_file_to_matrix::<PIXELS_PER_IMG, N_PICTURES>(PATH_TO_DATA).unwrap();
        let one_hot_y = one_hot::<N_PICTURES>(&y_matrix);

        let mut weights =
            match is_path_exist(&path_to_weights) {
                true => {
                    println!("Loading weights...");
                    load_weights::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>(&path_to_weights)
                }
                false => {
                    println!("Initializing weights...");
                    init_weights::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>()
                }
            };
        println!("Weights are set!");

        for epoch_n in 1..(TOTAL_EPOCHS + 1) {
            let start = std::time::Instant::now();

            let params = forward_pass(&x_matrix, &weights);

            let d =
                backward_pass::<PIXELS_PER_IMG, N_PICTURES, NEURONS_IN_HIDDEN_LAYER>(
                    &x_matrix, &one_hot_y,
                    &weights, &params,
                );

            update_params::<NEURONS_IN_HIDDEN_LAYER, PIXELS_PER_IMG>(
                &mut weights, &d, LR,
            );

            if epoch_n % 100 == 0 {
                save_weights(&weights, &path_to_weights).expect("Unable to save weights");
                println!("Weights saved!");
            }
            if epoch_n % 10 == 0 || epoch_n == 1 {
                println!("Accuracy: {:.6}", get_accuracy::<N_PICTURES>(&get_predictions::<N_PICTURES>(&params.a3), &y_matrix));
                println!("Epoch: {} : {:.2?}", epoch_n, start.elapsed());
            }
        }
    }).unwrap();

    handler.join().unwrap();
}
