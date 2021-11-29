#![allow(non_snake_case)]
use crate::bounds::Bounds1;
use crate::lp::LinearSolution;
use crate::NNVFloat;
use grb::constr::IneqExpr;
use grb::expr::{
    Expr::{Constant, Linear},
    LinExpr,
};
use grb::prelude::{add_var, attr, param, ConstrSense, Model, Status};
use grb::Env;
use grb::VarType::Continuous;
use ndarray::{Array1, ArrayView1};

/// # Panics
pub fn solve<'a, I, T: 'a + NNVFloat>(
    A: I,
    b: ArrayView1<T>,
    var_coeffs: ArrayView1<T>,
    var_bounds: &Bounds1<T>,
) -> LinearSolution
where
    I: IntoIterator<Item = ArrayView1<'a, T>>,
{
    // Create model
    let mut env = Env::empty().unwrap();
    env.set(param::OutputFlag, 0).unwrap();
    env.set(param::LogFile, "".to_string()).unwrap();
    let env = env.start().unwrap();
    let mut model = Model::with_env("", &env).unwrap();

    // Add variables
    let vars: Vec<_> = var_coeffs
        .iter()
        .zip(var_bounds.bounds_iter())
        .enumerate()
        .map(|(i, (c, b))| {
            add_var!(model, Continuous, obj: (*c).into(), name: &format!("v{}", i), bounds: b[[0]].into()..b[[1]].into()).unwrap()
        })
        .collect();

    // Add constraints
    let ineqs = A.into_iter().zip(b.iter()).map(|(A_i, b_i)| {
        let sense = ConstrSense::Less;
        let mut lhs = LinExpr::new();
        A_i.into_iter().zip(vars.iter()).for_each(|(A_ij, v)| {
            lhs.add_term((*A_ij).into(), *v);
        });

        let lhs = Linear(lhs);
        let rhs = Constant((*b_i).into());
        IneqExpr { lhs, sense, rhs }
    });
    let _constrs: Vec<_> = (0..b.len())
        .map(|i| format!("r{}", i))
        .zip(ineqs)
        .map(|(name, constr)| model.add_constr(&name, constr))
        .collect();

    // Optimize and get status
    model.set_param(param::InfUnbdInfo, 1).unwrap();
    model.optimize().unwrap();
    let status = model.status().unwrap();
    let output = match status {
        Status::Optimal => LinearSolution::Solution(
            Array1::from_vec(model.get_obj_attr_batch(attr::X, vars).unwrap()),
            model.get_attr(attr::ObjVal).unwrap(),
        ),
        Status::Infeasible => LinearSolution::Infeasible,
        Status::Unbounded => {
            let ray = Array1::from_vec(model.get_obj_attr_batch(attr::UnbdRay, vars).unwrap());
            LinearSolution::Unbounded(ray)
        }
        _ => panic!("Gurobi failed to solve!"),
    };
    output
}
