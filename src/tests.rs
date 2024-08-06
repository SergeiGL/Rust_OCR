use std::fs;
use super::*;
use fs::remove_file;
#[test]
fn is_path_exist_0(){
    assert_eq!(is_path_exist("src"), true)
}

#[test]
fn is_path_exist_1(){
    assert_eq!(is_path_exist("src/main.rs"), true)
}

#[test]
fn is_path_exist_2(){
    assert_eq!(is_path_exist("src/asd73577asdsda52sasd/main.rs"), false)
}

#[test]
fn is_path_exist_3(){
    assert_eq!(is_path_exist("src/asd73577asdsda52sasd"), false)
}

#[test]
fn transpose_0() {
    assert_eq!(
        transpose(
            &vec![
                vec![0.0_f32, 0.0_f32],
                vec![0.0_f32, 0.0_f32]
            ])
        ,
            vec![
                vec![0.0_f32, 0.0_f32],
                vec![0.0_f32, 0.0_f32]
            ]
    );
}

#[test]
fn transpose_1() {
    assert_eq!(
        transpose(
            &vec![
                vec![1.1_f32, 2.2_f32],
                vec![3.3_f32, 4.4_f32]
            ])
        ,
            vec![
                vec![1.1_f32, 3.3_f32],
                vec![2.2_f32, 4.4_f32]
            ]
    );
}

#[test]
fn transpose_2() {
    assert_eq!(
        transpose(&vec![vec![1.0_f32, 2.0_f32], vec![3.9_f32, 4.0_f32], vec![5.0_f32, 6.0_f32]]),
        vec![vec![1.0_f32, 3.9_f32, 5.0_f32], vec![2.0_f32, 4.0_f32, 6.0_f32]]
    );
}

#[test]
fn transpose_3() {
    assert_eq!(
        transpose(&vec![vec![1.0123_f32, 2.1245_f32]]),
        vec![vec![1.0123_f32], vec![2.1245_f32]]
    );
}

#[test]
fn transpose_4() {
    assert_eq!(
        transpose(&vec![vec![-132.42123_f32], vec![-124.14245_f32]]),
        vec![vec![-132.42123_f32, -124.14245_f32]]
    );
}

#[test]
fn one_hot_0() {
    assert_eq!(
        one_hot(&vec![1_usize, 0_usize, 9_usize]),
        vec![
            vec![0.0_f32, 1.0_f32, 0.0_f32],
            vec![1.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 1.0_f32]
        ]
    );
}

#[test]
fn one_hot_1() {
    assert_eq!(
        one_hot(&vec![0_usize, 1_usize, 2_usize, 3_usize, 4_usize, 5_usize, 6_usize, 7_usize, 8_usize, 9_usize, 0_usize]),
        vec![
            vec![1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32],
            vec![0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32],
            vec![0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32]
        ]

    );
}

#[test]
fn lin_cmb_0() {
    assert_eq!(linear_combination(&vec![vec![39_f32], vec![74_f32], vec![142_f32]],
        &vec![
            vec![1_f32, 2_f32, 3_f32, 4_f32],
            vec![5_f32, 6_f32, 7_f32, 8_f32],
            vec![9_f32, 10_f32, 11_f32, 12_f32]],
        &vec![
            vec![19_f32, 21_f32, 33_f32],
            vec![42_f32, 55_f32, 67_f32],
            vec![72_f32, 81_f32, 91_f32],
            vec![76_f32, 82_f32, 90_f32]])
        ,
        vec![
            vec![662_f32, 741_f32, 839_f32],
            vec![1533_f32, 1732_f32, 1998_f32],
            vec![2437_f32, 2756_f32, 3190_f32]],
    );
}

#[test]
fn lin_cmb_1() {
    assert_eq!(linear_combination(&vec![vec![0_f32], vec![1.1_f32], vec![-1_f32], vec![5_f32]],
        &vec![
            vec![0_f32, 2_f32, 3_f32],
            vec![5_f32, 0_f32, 7_f32],
            vec![9_f32, 10_f32, 0_f32],
            vec![0_f32, 0.3_f32, 1_f32]],
        &vec![
            vec![-1.1_f32, 1_f32, -1_f32, -6_f32],
            vec![2_f32, -5_f32, -7_f32, -2_f32],
            vec![2_f32, 0_f32, 0_f32, 0_f32]])
        ,
        vec![
            vec![10_f32, -10_f32, -14_f32, -4_f32],
            vec![9.6_f32, 6.1_f32, -3.9_f32, -28.9_f32],
            vec![9.099999_f32, -42_f32, -80_f32, -75_f32],
            vec![7.6_f32, 3.5_f32, 2.8999999_f32, 4.4_f32]],
    );
}

#[test]
fn relu_0() {
    assert_eq!(
        relu(&vec![
            vec![-1.1_f32, -0.1_f32, 1_f32, -6_f32],
            vec![0.2_f32, -5_f32, -0.7_f32, -2_f32],
            vec![2_f32, 0_f32, 0.01_f32, -0.01_f32]])
    ,
        vec![
            vec![0_f32, 0_f32, 1_f32, 0_f32],
            vec![0.2_f32, 0_f32, 0_f32, 0_f32],
            vec![2_f32, 0_f32, 0.01_f32, 0_f32]]
    )
}

#[test]
fn softmax_0() {
    let mut mtrix = vec![
            vec![-1.1_f32, -0.1_f32, 1_f32, -6_f32],
            vec![0.2_f32, -5_f32, -0.7_f32, -2_f32],
            vec![2_f32, 0_f32, 0.01_f32, -0.01_f32]];

    softmax(&mut mtrix);
    assert_eq!(mtrix,
        vec![
            vec![0.03722004, 0.47334647, 0.6433929, 0.002197741],
            vec![0.13657136, 0.0035248138, 0.11753728, 0.11999257],
            vec![0.8262086, 0.52312875, 0.23906982, 0.87780964]
        ]
    );
}

#[test]
fn mtrix_np_sum_0(){
    assert_eq!(
        mtrix_np_sum(&vec![vec![1_f32]]), vec![vec![1_f32]]
    );
}

#[test]
fn mtrix_np_sum_1() {
    assert_eq!(
        mtrix_np_sum(&vec![vec![1_f32, -1_f32, 3_f32], vec![3_f32, 2_f32, 3_f32]]),
        vec![vec![3_f32], vec![8_f32]]
    );
}

#[test]
fn mtrix_np_sum_2() {
    assert_eq!(
        mtrix_np_sum(&vec![vec![1.5, 2.5], vec![-1.0, 1.0], vec![0.5, -0.5]]),
        vec![vec![4.0], vec![0.0], vec![0.0]]
    );
}

#[test]
fn mtrix_np_sum_3() {
    assert_eq!(
        mtrix_np_sum(&vec![vec![0_f32; 3]; 3]),
        vec![vec![0_f32], vec![0_f32], vec![0_f32]]
    );
}



#[test]
fn mtrix_scale_by_c_0(){
    let mut mtrix =
        vec![
            vec![1_f32, 0_f32, -1_f32, 0.01_f32, -0.01_f32],
            vec![0.0_f32, -11_f32, -35.7_f32, -0.11_f32, 10.234_f32],
            vec![1.5_f32, 72_f32, -0.235_f32, -124.36_f32, 83_f32],
            vec![-10.6_f32, -5.5_f32, -1.6_f32, -7_f32, 0.03_f32],
            ];
    mtrix_scale_by_c(-1.23456_f32, &mut mtrix);
    assert_eq!(mtrix,
        vec![
            vec![-1.23456_f32, 0_f32, 1.23456_f32, -0.012345600000000002_f32, 0.012345600000000002_f32],
            vec![0.0_f32, 13.580160000000001_f32, 44.073792000000005_f32, 0.13580160000000002_f32, -12.634487040000002_f32],
            vec![-1.8518400000000002_f32, -88.88832000000001_f32, 0.29012161_f32, 153.5298816_f32, -102.46848000000001_f32],
            vec![13.086336000000001_f32, 6.790080000000001_f32, 1.9752960000000002_f32, 8.64192_f32, -0.0370368_f32],
        ]
    );
}


#[test]
fn save_and_load_weights_0(){
    let path = "weights/test_weights.bin";

    let w_in = vec![vec![1.1_f32, 2.2_f32], vec![13.3_f32, -4.4_f32]];
    let w_h = vec![vec![5.5_f32, -6.6_f32, 14_f32], vec![17.7_f32, 8.8_f32, 14_f32]];
    let w_out = vec![vec![5.5_f32, -6.6_f32, 14_f32], vec![17.7_f32, 8.8_f32, 14_f32]];

    let b_in = vec![vec![-1.1_f32, -12.2_f32], vec![3.3_f32, -4.4_f32]];
    let b_h = vec![vec![-5.5_f32, 16.6_f32, 14_f32], vec![7.7_f32, 8.8_f32, 14_f32]];
    let b_out = vec![vec![-5.5_f32, 16.6_f32, 14_f32], vec![7.7_f32, 8.8_f32, 14_f32]];

    save_weights(&b_in, &w_in, &b_h, &w_h, &b_out, &w_out, path).expect("Unable to save weights");

    let (b_in_l, w_in_l, b_h_l, w_h_l, b_out_l, w_out_l)
        = load_weights(path).unwrap();

    assert_eq!(w_in, w_in_l);
    assert_eq!(w_h, w_h_l);
    assert_eq!(w_out, w_out_l);
    assert_eq!(b_in, b_in_l);
    assert_eq!(b_h, b_h_l);
    assert_eq!(b_out, b_out_l);
}


#[test]
fn canvas_to_matrix_0() {
    assert_eq!(
        canvas_to_matrix(&[0x000000_u32, 0xFFFFFF_u32, 0x000000_u32, 0xFFFFFF_u32, 0x000000_u32, 0x000000_u32, 0xFFFFFF_u32, 0xFFFFFF_u32])
        ,
        vec![
            vec![1_f32],
            vec![0_f32],
            vec![1_f32],
            vec![0_f32],
            vec![1_f32],
            vec![1_f32],
            vec![0_f32],
            vec![0_f32],
        ]
    );
}


#[test]
fn save_load_read_0(){
    let basedir = "data_test/";
    create_fldrs_if_not_exist("", vec![String::from(basedir)]);

    let test_dir_postfix = "test_0/";
    create_fldrs_if_not_exist(basedir, vec![String::from(test_dir_postfix)]);

    create_fldrs_if_not_exist(basedir, vec![String::from(test_dir_postfix)]);

    let path_to_data_dir = &format!("{}{}",basedir, test_dir_postfix);
    create_fldrs_if_not_exist(path_to_data_dir, (0..10).map(|i| i.to_string()).collect());

    let path_to_data_file= &format!("{}{}/{}", path_to_data_dir, 9, 14);
    let canvas = [0x000000_u32, 0x000000_u32, 0xFFFFFF_u32, 0x000000_u32, 0xFFFFFF_u32, 0x000000_u32, 0x000000_u32, 0xFFFFFF_u32, 0xFFFFFF_u32];

    save_canvas(&canvas, path_to_data_file).unwrap();

    let (x_matrix, _y) = read_file_to_matrix(path_to_data_dir, 1, canvas.len()).unwrap();

    println!("Removing {}", path_to_data_file);
    remove_file(path_to_data_file).unwrap();

    assert_eq!(
        canvas_to_matrix(&canvas), x_matrix
    );
}

#[test]
fn save_load_read_1() {
    let basedir = "data_test/";
    create_fldrs_if_not_exist("", vec![String::from(basedir)]);

    let test_dir_postfix = "test_1/";
    create_fldrs_if_not_exist(basedir, vec![String::from(test_dir_postfix)]);

    let canvas = [0x000000_u32; 20];

    let path_to_data_dir = &format!("{}{}", basedir, test_dir_postfix);
    create_fldrs_if_not_exist(path_to_data_dir, (0..10).map(|i| i.to_string()).collect());

    let paths_to_data: Vec<String> = (0..10).map(|i| format!("{}{}/{}", path_to_data_dir, i, i+1)).collect();

    for (path_n, path_to_data_file) in paths_to_data.iter().enumerate(){
        let mut canvas_modif = canvas.clone();
        canvas_modif[path_n] = 0xFFFFFF_u32;

        save_canvas(&canvas_modif, &path_to_data_file).unwrap();
        let (x_matrix, _y) = read_file_to_matrix(path_to_data_dir, 1, canvas.len()).unwrap();

        println!("Removing {}", path_to_data_file);
        remove_file(path_to_data_file).unwrap();

        assert_eq!(
            canvas_to_matrix(&canvas_modif), x_matrix
        );
    }
}


#[test]
fn clear_canvas_0() {
    let mut canvas = [0x000000_u32, 0x000000_u32, 0xFFFFFF_u32, 0x000000_u32, 0xFFFFFF_u32, 0x000000_u32, 0x000000_u32, 0xFFFFFF_u32, 0xFFFFFF_u32];
    clear_canvas(&mut canvas);

    assert_eq!(
        canvas, [0xFFFFFF_u32; 9]
    );

}

#[test]
fn mtrix_minus_in_place_0(){
    let mut left = vec![
        vec![4.0, 6.0, -4.4],
        vec![7.0, 8.0, -5.0],
    ];
    let right = vec![
        vec![0.0, 2.0, -0.4],
        vec![3.0, 4.0, -1.0],
    ];
    mtrix_minus_in_place(&mut left, &right);

    assert_eq!(left,
       vec![
            vec![4.0, 4.0, -4.0],
            vec![4.0, 4.0, -4.0],
    ]);
}

#[test]
fn mtrix_minus_in_place_1() {
    let mut left = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    ];
    let right = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    ];
    mtrix_minus_in_place(&mut left, &right);

    assert_eq!(left,
        vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
    ]);
}



#[test]
fn mtrix_dot_0(){
    let left = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
    let right = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
    let expected = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        assert_eq!(mtrix_dot(&left, &right), expected);
}

#[test]
fn mtrix_dot_1(){
    let left = vec![vec![2.0]];
    let right = vec![vec![3.0]];
    let expected = vec![vec![6.0]];
    assert_eq!(mtrix_dot(&left, &right), expected);
}

#[test]
fn mtrix_dot_2(){
    let left = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ];
    let right = vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];
    let expected = vec![
        vec![19.0, 22.0],
        vec![43.0, 50.0],
    ];
    assert_eq!(mtrix_dot(&left, &right), expected);
}

#[test]
fn mtrix_dot_3() {
    let left = vec![
        vec![1.1, -2.2],
        vec![-3.3, 4.4],
    ];
    let right = vec![
        vec![-1.0, 0.0],
        vec![0.0, -1.0],
    ];
    let expected = vec![
        vec![-1.1, 2.2],
        vec![3.3, -4.4],
    ];
    assert_eq!(mtrix_dot(&left, &right), expected);
}

#[test]
#[should_panic]
fn mtrix_dot_4() {
    let left = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ];
    let right = vec![
        vec![5.0, 6.0, 7.0],
    ];
    mtrix_dot(&left, &right);
}


#[test]
fn mtrix_mult_0() {
    let left = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ];
    let right = vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];
    let expected = vec![
        vec![5.0, 12.0],
        vec![21.0, 32.0],
    ];
    assert_eq!(mtrix_mult(&left, &right), expected);
}

#[test]
fn mtrix_mult_1() {
    let left = vec![
        vec![-2.2, 0.1],
        vec![0.1, -2.2],
    ];
    let right = vec![
        vec![2.0, 0.0],
        vec![0.0, 2.0],
    ];
    let expected = vec![
        vec![-4.4, 0.0],
        vec![0.0, -4.4],
    ];
    assert_eq!(mtrix_mult(&left, &right), expected);
}

#[test]
fn mtrix_mult_2() {
    let left = vec![
        vec![-1.0, -2.0],
        vec![-3.0, -4.0],
    ];
    let right = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
    ];
    let expected = vec![
        vec![-1.0, -4.0],
        vec![-9.0, -16.0],
    ];
    assert_eq!(mtrix_mult(&left, &right), expected);
}

#[test]
fn mtrix_mult_3() {
    let left = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0]
    ];
    let right = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0]
    ];
    let expected = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0]
    ];
    assert_eq!(mtrix_mult(&left, &right), expected);
}

#[test]
fn mtrix_mult_4() {
    let left = vec![
        vec![1.5, -2.5],
        vec![-3.5, 4.5],
    ];
    let right = vec![
        vec![1.5, -2.5],
        vec![-3.5, 4.5],
    ];
    let expected = vec![
        vec![2.25, 6.25],
        vec![12.25, 20.25],
    ];
    assert_eq!(mtrix_mult(&left, &right), expected);
}

#[test]
fn relu_deriv_0() {
    let mtrix = vec![
        vec![-1.1, -2.2, 3.3],
        vec![4.4, -5.5, 6.6]
    ];
    let expected = vec![
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0]
    ];
    assert_eq!(relu_deriv(&mtrix), expected);
}

#[test]
fn relu_deriv_1() {
    let mtrix = vec![
        vec![1.0],
        vec![4.0]
    ];
    let expected = vec![
        vec![1.0],
        vec![1.0]
    ];
    assert_eq!(relu_deriv(&mtrix), expected);
}

#[test]
fn relu_deriv_2() {
    let mtrix = vec![
        vec![-1.0],
        vec![-4.0]
    ];
    let expected = vec![
        vec![0.0],
        vec![0.0]
    ];
    assert_eq!(relu_deriv(&mtrix), expected);
}

#[test]
fn relu_deriv_3() {
    let mtrix = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0]
    ];
    let expected = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0]
    ];
    assert_eq!(relu_deriv(&mtrix), expected);
}


#[test]
fn get_predictions_0() {
    let input = vec![
        vec![-0.2, 0.3, 0.1],
        vec![0.4, -0.1, 0.5],
        vec![0.6, 0.2, -0.9]
    ];
    let expected_output = vec![2, 0, 1];
    assert_eq!(get_predictions(&input), expected_output);
}

#[test]
fn get_predictions_1() {
    let input = vec![
        vec![1.0, 0.2, 0.3],
        vec![0.1, 0.8, 0.4],
        vec![0.5, 0.1, 0.9],
        vec![0.6, 0.2, -0.1]
    ];
    let expected_output = vec![0, 1, 2];
    assert_eq!(get_predictions(&input), expected_output);
}

#[test]
fn get_predictions_2() {
    let input = vec![
        vec![-0.5, -0.5, -0.5],
        vec![-0.5, -0.5, -0.5],
        vec![-0.5, -0.5, -0.5]
    ];
    let expected_output = vec![0, 0, 0];
    assert_eq!(get_predictions(&input), expected_output);
}

#[test]
fn get_predictions_3() {
    let input = vec![
        vec![-0.1],
        vec![-0.02],
        vec![-0.3]
    ];
    let expected_output = vec![1];
    assert_eq!(get_predictions(&input), expected_output);
}

#[test]
fn get_predictions_4() {
    let input = vec![
        vec![0.8, 0.3, -0.5, 0.7],
        vec![0.2, 0.9, 0.4, 0.6],
        vec![-0.1, -0.8, 0.6, 0.5]
    ];
    let expected_output = vec![0, 1, 2, 0];
    assert_eq!(get_predictions(&input), expected_output);
}

#[test]
fn get_accuracy_0() {
    let predictions = vec![1, 2, 3, 4];
    let y_matrix = vec![1, 2, 3, 4];
    assert_eq!(get_accuracy(&predictions, &y_matrix), 1.0);
}

#[test]
fn get_accuracy_1() {
    let predictions = vec![1, 0, 0, 0];
    let y_matrix = vec![1, 0, 0, 0];
    assert_eq!(get_accuracy(&predictions, &y_matrix), 1.0);
}

#[test]
fn get_accuracy_2() {
    let predictions = vec![1, 2, 3, 4, 5];
    let y_matrix = vec![6, 7, 8, 9, 10];
    assert_eq!(get_accuracy(&predictions, &y_matrix), 0.0);
}

#[test]
fn get_accuracy_3() {
    let predictions = vec![0, 1, 2, 3, 4];
    let y_matrix = vec![4, 3, 2, 1, 0];
    assert_eq!(get_accuracy(&predictions, &y_matrix), 0.2);
}

#[test]
fn get_accuracy_4() {
    let predictions = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let y_matrix =    vec![0, 1, 3, 2, 4, 5, 7, 6, 8, 0];
    assert_eq!(get_accuracy(&predictions, &y_matrix), 0.5);
}

