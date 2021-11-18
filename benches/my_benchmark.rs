use criterion::{black_box, criterion_group, criterion_main, Criterion};
use env_logger::Builder;
use env_logger::Env;
use env_logger::Target;
use log::info;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use nnv_rs::affine::Affine;
use nnv_rs::affine::Affine2;
use nnv_rs::asterism::Asterism;
use nnv_rs::bounds::Bounds;
use nnv_rs::constellation::Constellation;
use nnv_rs::deeppoly::*;
use nnv_rs::dnn::Layer;
use nnv_rs::dnn::DNN;
use nnv_rs::star::Star2;
use nnv_rs::star_node::StarNode;
use pprof::criterion::{Output, PProfProfiler};
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::process::exit;
use std::process::Command;

fn bench(c: &mut Criterion) {
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let mut git_hash = String::from_utf8(output.stdout).unwrap();
    git_hash.pop();
    git_hash.push_str(".log");
    println!("{}", git_hash);
    let mut buffer = BufWriter::new(File::create(git_hash).unwrap());

    let env = Env::default()
        .filter_or("MY_LOG_LEVEL", "trace")
        .write_style_or("MY_LOG_STYLE", "always");
    let mut builder = Builder::from_env(env);
    let targeted_builder = builder.target(Target::Pipe(Box::new(buffer)));
    println!("builder {:?}", targeted_builder);
    targeted_builder.init();
    println!("builder {:?}", targeted_builder);
    info!("Test!");
    exit(0);

    let mut rng = rand::thread_rng();
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
    let mut root_node = StarNode::default(root_star.clone(), None);
    let loc = Array1::zeros(input_size);
    let scale = Array2::from_diag(&Array1::ones(input_size));
    c.bench_function("starnode::get_output_bounds from root", |b| {
        b.iter(|| root_node.get_output_bounds(&dnn, &|x| (x.lower()[[0]], x.upper()[[0]])))
    });

    let mut group = c.benchmark_group("constellation");
    group.bench_function("constellation::sample_safe_star::random_net@0", |b| {
        b.iter(|| {
            let mut constellation = Constellation::new(
                root_star.clone(),
                dnn.clone(),
                Some(bounds.clone()),
                loc.clone(),
                scale.clone(),
            );
            let mut asterism = Asterism::new(&mut constellation, 0.);
            asterism.sample_safe_star(1, &mut rng, 100, 20, None);
        })
    });
    group.bench_function("constellation::sample_safe_star::random_net@-5", |b| {
        b.iter(|| {
            let mut constellation = Constellation::new(
                root_star.clone(),
                dnn.clone(),
                Some(bounds.clone()),
                loc.clone(),
                scale.clone(),
            );
            let mut asterism = Asterism::new(&mut constellation, -5.);
            asterism.sample_safe_star(1, &mut rng, 100, 20, None);
        })
    });
    group.bench_function("constellation::sample_safe_star::random_net@-100", |b| {
        b.iter(|| {
            let mut constellation = Constellation::new(
                root_star.clone(),
                dnn.clone(),
                Some(bounds.clone()),
                loc.clone(),
                scale.clone(),
            );
            let mut asterism = Asterism::new(&mut constellation, -100.);
            asterism.sample_safe_star(1, &mut rng, 100, 20, None);
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench
}
criterion_main!(benches);
