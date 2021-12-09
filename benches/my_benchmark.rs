use criterion::{criterion_group, criterion_main, Criterion};
use env_logger::Builder;
use env_logger::Env;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Ix2;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use nnv_rs::affine::Affine2;
use nnv_rs::asterism::Asterism;
use nnv_rs::bounds::Bounds;
use nnv_rs::constellation::Constellation;
use nnv_rs::dnn::Layer;
use nnv_rs::dnn::DNN;
use nnv_rs::star::Star2;
use nnv_rs::star_node::StarNode;
use nnv_rs::tensorshape::TensorShape;
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::process::Command;
use std::time::Duration;

fn build_dnn<R: Rng>(input_size: usize, rng: &mut R) -> DNN<f64> {
    let dense1 = Layer::new_dense(Affine2::new(
        Array::random_using((64, input_size), Normal::new(0., 1.).unwrap(), rng),
        Array::random_using(64, Normal::new(0., 1.).unwrap(), rng),
    ));
    let relu1 = Layer::new_relu(64);
    let dense2 = Layer::new_dense(Affine2::new(
        Array::random_using((64, 64), Normal::new(0., 1.).unwrap(), rng),
        Array::random_using(64, Normal::new(0., 1.).unwrap(), rng),
    ));
    let relu2 = Layer::new_relu(64);
    let dense_out = Layer::new_dense(Affine2::new(
        Array::random_using((1, 64), Normal::new(0., 1.).unwrap(), rng),
        Array::random_using(1, Normal::new(0., 1.).unwrap(), rng),
    ));
    DNN::new(vec![dense1, relu1, dense2, relu2, dense_out])
}

fn bench(c: &mut Criterion) {
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let mut git_hash = String::from_utf8(output.stdout).unwrap();
    git_hash.pop();
    println!("Git hash: {}", git_hash);

    let env = Env::default();
    let mut builder = Builder::from_env(env);
    builder.init();

    let mut rng = Pcg64::seed_from_u64(69);
    let input_size = 68;

    let bounds = Bounds::new(
        (Array1::ones(input_size) * -3.5).view(),
        (Array1::ones(input_size) * 3.5).view(),
    );
    let root_star =
        Star2::default(&TensorShape::new(vec![Some(input_size)])).with_input_bounds(bounds.clone());
    let mut root_node = StarNode::default(root_star.clone(), None);
    let loc = Array1::zeros(input_size);
    let scale = Array2::from_diag(&Array1::ones(input_size));

    c.bench_function("starnode::get_output_bounds from root", |b| {
        b.iter_batched(
            || build_dnn(input_size, &mut rng),
            |dnn: DNN<f64>| {
                root_node.get_output_bounds(&dnn, &|x| (x.lower()[[0]], x.upper()[[0]]))
            },
            criterion::BatchSize::SmallInput,
        )
    });

    let mut group = c.benchmark_group("constellation");
    group.warm_up_time(Duration::from_secs(10));
    group.measurement_time(Duration::from_secs(300));
    let mut local_rng = Pcg64::from_rng(&mut rng).unwrap();
    group.bench_function("constellation::sample_safe_star::random_net@0", |b| {
        b.iter_batched(
            || {
                let dnn = build_dnn(input_size, &mut local_rng);
                Constellation::new(root_star.clone(), dnn, loc.clone(), scale.clone())
            },
            |mut constellation: Constellation<f64, Ix2>| {
                let mut asterism = Asterism::new(&mut constellation, 0.);
                asterism.sample_safe_star(1, &mut rng, 100, 20, None, 1e-4);
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("constellation::sample_safe_star::random_net@-10", |b| {
        b.iter_batched(
            || {
                let dnn = build_dnn(input_size, &mut local_rng);
                Constellation::new(root_star.clone(), dnn, loc.clone(), scale.clone())
            },
            |mut constellation: Constellation<f64, Ix2>| {
                let mut asterism = Asterism::new(&mut constellation, -10.);
                asterism.sample_safe_star(1, &mut rng, 100, 20, None, 1e-4);
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.sample_size(10);
    group.bench_function("constellation::sample_safe_star::random_net@-100", |b| {
        b.iter_batched(
            || {
                let dnn = build_dnn(input_size, &mut local_rng);
                Constellation::new(root_star.clone(), dnn, loc.clone(), scale.clone())
            },
            |mut constellation: Constellation<f64, Ix2>| {
                let mut asterism = Asterism::new(&mut constellation, -100.);
                asterism.sample_safe_star(1, &mut rng, 100, 20, None, 1e-4);
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench
}
criterion_main!(benches);
