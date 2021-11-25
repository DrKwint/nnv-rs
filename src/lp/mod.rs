use ndarray::Array1;

cfg_if::cfg_if! {
    if #[cfg(feature = "lp_gurobi")] {
        mod gurobi;
        pub use gurobi::solve;
    } else if #[cfg(feature = "lp_coincbc")] {
        mod coincbc;
        pub use coincbc::solve;
    } else {
        compile_error!("Must enable one of \"lp_{{gurobi,coincbc}}\"");
    }
}

#[derive(Debug)]
pub enum LinearSolution {
    Solution(Array1<f64>, f64),
    Infeasible,
    Unbounded(Array1<f64>),
}
