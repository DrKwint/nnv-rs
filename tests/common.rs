use ndarray::Array;
use ndarray::Dim;
use nnv_rs::affine::Affine2;
use nnv_rs::dnn::dense::Dense;
use nnv_rs::dnn::dnn::DNN;
use nnv_rs::dnn::relu::ReLU;
use rand::Rng;

pub fn affine2(in_dim: &usize, out_dim: &usize) -> Affine2 {
    let mut rng = rand::thread_rng();
    let basis_vec = (0..(in_dim * out_dim)).map(|_| rng.gen()).collect();
    let shift_vec: Vec<f64> = (0..*out_dim).map(|_| rng.gen()).collect();
    let shape = Dim([*out_dim, *in_dim]);

    let basis = Array::from_shape_vec(shape, basis_vec).unwrap();
    let shift = Array::from(shift_vec);
    Affine2::new(basis, shift)
}

pub fn make_dnn(shape: &Vec<usize>, num_layers: &usize) -> DNN {
    assert_eq!(shape.len(), 2);
    let mut dnn = DNN::default();

    (0..*num_layers).into_iter().for_each(|_| {
        dnn.add_layer(Box::new(Dense::new(affine2(&shape[0], &shape[1]))));
        dnn.add_layer(Box::new(ReLU::new(shape[1])));
    });

    dnn
}
