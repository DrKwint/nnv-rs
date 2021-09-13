use ndarray::Array1;
use ndarray::Ix1;
use nnv_rs_py::bounds::Bounds;
use nnv_rs_py::deeppoly::*;
use nnv_rs_py::dnn::DNNIndex;
use nnv_rs_py::dnn::DNNIterator;

mod common;

#[test]
fn test_deeppoly_correctness() {
    let dnn = common::make_dnn(&vec![2, 2], &1);
    let lower_bounds = Array1::ones(2) * -20.;
    let upper_bounds = Array1::ones(2) * 20.;
    let input_bounds = Bounds::new(lower_bounds, upper_bounds);

    let concrete_input = input_bounds.sample_uniform(0u64);
    let output_bounds = deep_poly(input_bounds, DNNIterator::new(&dnn, DNNIndex::default()));
    let concrete_output = dnn
        .forward(concrete_input.into_dyn())
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(
        output_bounds.is_member(&concrete_output.view()),
        "\n\nConcrete output: {}\nOutput bounds: {}\n\n",
        concrete_output,
        output_bounds
    );
}