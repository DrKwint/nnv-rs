use ndarray::Array;
use ndarray::Dim;
use nnv_rs_py::affine::Affine2;
use nnv_rs_py::dnn::{Layer, DNN};
use rand::Rng;

pub fn affine2(in_dim: &usize, out_dim: &usize) -> Affine2<f64> {
    let mut rng = rand::thread_rng();
    let basis_vec = (0..(in_dim * out_dim)).map(|_| rng.gen()).collect();
    let shift_vec: Vec<f64> = (0..*out_dim).map(|_| rng.gen()).collect();
    let shape = Dim([*out_dim, *in_dim]);

    let basis = Array::from_shape_vec(shape, basis_vec).unwrap();
    let shift = Array::from(shift_vec);
    Affine2::new(basis, shift)
}

pub fn make_dnn(shape: &Vec<usize>, num_layers: &usize) -> DNN<f64> {
    assert_eq!(shape.len(), 2);
    let mut dnn = DNN::default();

    (0..*num_layers).into_iter().for_each(|_| {
        dnn.add_layer(Layer::new_dense(affine2(&shape[0], &shape[1])));
        dnn.add_layer(Layer::new_relu(shape[1]));
    });

    dnn
}
