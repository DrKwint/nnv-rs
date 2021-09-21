use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use nnv_rs_py::affine::Affine;
use nnv_rs_py::affine::Affine2;
use nnv_rs_py::constellation::Constellation;
use nnv_rs_py::deeppoly::*;
use nnv_rs_py::star::Star2;
use nnv_rs_py::Bounds;
use nnv_rs_py::Layer;
use nnv_rs_py::StarNode;
use nnv_rs_py::DNN;

fn criterion_benchmark(c: &mut Criterion) {
    let input_size = 68;
    let dense1 = Layer::new_dense(Affine2::new(
        Array::random((64, input_size), Normal::new(0., 1.).unwrap()),
        Array::random(64, Normal::new(0., 1.).unwrap()),
    ));
    let relu1 = Layer::new_relu(64);
    let dense2 = Layer::new_dense(Affine2::new(
        Array::random((64, 64), Normal::new(0., 1.).unwrap()),
        Array::random(64, Normal::new(0., 1.).unwrap()),
    ));
    let relu2 = Layer::new_relu(64);
    let dense_out = Layer::new_dense(Affine2::new(
        Array::random((1, 64), Normal::new(0., 1.).unwrap()),
        Array::random(1, Normal::new(0., 1.).unwrap()),
    ));
    let dnn = DNN::new(vec![dense1, relu1, dense2, relu2, dense_out]);

    let bounds = Bounds::new(
        (Array1::ones(input_size) * -3.5).view(),
        (Array1::ones(input_size) * 3.5).view(),
    );
    let root_star = Star2::default(&dnn.input_shape()).with_input_bounds(bounds.clone());
    let mut root_node = StarNode::default(root_star.clone());
    let loc = Array1::zeros(input_size);
    let scale = Array2::from_diag(&Array1::ones(input_size));
    c.bench_function("starnode::get_output_bounds from root", |b| {
        b.iter(|| root_node.get_output_bounds(&dnn, &|x| (x.lower()[[0]], x.upper()[[0]])))
    });
    c.bench_function("constellation::sample_safe_star", |b| {
        b.iter(|| {
            let mut constellation =
                Constellation::new(root_star.clone(), dnn.clone(), Some(bounds.clone()));
            constellation.sample_safe_star(&loc, &scale, 1., 100, 20)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
