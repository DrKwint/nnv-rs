#[macro_use]
extern crate criterion;
use criterion::{black_box, BenchmarkId, Criterion};

use criterion::profiler::Profiler;
use pprof::criterion::{Output, PProfProfiler};
use pprof::protos::Message;
use pprof::ProfilerGuard;
use std::fs::create_dir_all;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Thanks to the example provided by @jebbow in his article
// https://www.jibbow.com/posts/criterion-flamegraphs/

#[cfg(unix)]
#[derive(Default)]
struct CpuProfiler<'a> {
    guard: Option<ProfilerGuard<'a>>,
}

#[cfg(unix)]
impl<'a> Profiler for CpuProfiler<'a> {
    fn start_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        create_dir_all(&benchmark_dir).unwrap();

        let guard = ProfilerGuard::new(100).unwrap();
        self.guard = Some(guard);
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        if let Ok(ref report) = self.guard.as_ref().unwrap().report().build() {
            let fg_file_name = benchmark_dir.join(format!("{}.svg", benchmark_id));
            let fg_file = File::create(fg_file_name).unwrap();
            report.flamegraph(fg_file).unwrap();

            let pb_file_name = benchmark_dir.join(format!("{}.pb", benchmark_id));
            let mut pb_file = File::create(pb_file_name).unwrap();
            let profile = report.pprof().unwrap();

            let mut content = Vec::new();
            profile.encode(&mut content).unwrap();
            pb_file.write_all(&content).unwrap();
        };

        self.guard = None;
    }
}

fn fibonacci(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn bench(c: &mut Criterion) {
    c.bench_function("Fibonacci", |b| b.iter(|| fibonacci(black_box(20))));
}

fn bench_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fibonacci Sizes");

    for s in &[1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(s), s, |b, s| {
            b.iter(|| fibonacci(black_box(*s)))
        });
    }
}

criterion_group! {
    name = benches;
    //config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    config = Criterion::default().with_profiler(CpuProfiler::default());
    targets = bench, bench_group
}
criterion_main!(benches);
