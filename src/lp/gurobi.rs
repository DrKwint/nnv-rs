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
use std::ops::Deref;

#[allow(dead_code)]
pub fn qsolve<'a, I, J>(
    _A: &I,
    _b: &J,
    _qvar_coeffs: ArrayView1<NNVFloat>,
    _var_coeffs: ArrayView1<NNVFloat>,
    _var_bounds: &Option<Bounds1>,
) -> LinearSolution
where
    I: IntoIterator<Item = ArrayView1<'a, NNVFloat>>,
    J: IntoIterator<Item = &'a NNVFloat>,
{
    todo!()
}

/// # Panics
pub fn solve<'a, I, J, Bounds1Ref: Deref<Target = Bounds1>>(
    A: I,
    b: J,
    var_coeffs: ArrayView1<NNVFloat>,
    var_bounds_opt: Option<Bounds1Ref>,
) -> LinearSolution
where
    I: IntoIterator<Item = ArrayView1<'a, NNVFloat>>,
    J: IntoIterator<Item = &'a NNVFloat>,
{
    // Create model
    let mut env = Env::empty().unwrap();
    env.set(param::OutputFlag, 0).unwrap();
    env.set(param::LogFile, "".to_string()).unwrap();
    let env = env.start().unwrap();
    let mut model = Model::with_env("", &env).unwrap();

    // Add variables
    let vars: Vec<_> = if let Some(var_bounds) = var_bounds_opt {
        var_coeffs
            .iter()
            .zip(var_bounds.bounds_iter())
            .enumerate()
            .map(|(i, (c, b))| {
                add_var!(model, Continuous, obj: *c, name: &format!("v{}", i), bounds: b[[0]]..b[[1]])
                    .unwrap()
            })
            .collect()
    } else {
        var_coeffs
            .iter()
            .enumerate()
            .map(|(i, c)| add_var!(model, Continuous, obj: *c, name: &format!("v{}", i)).unwrap())
            .collect()
    };

    // Add constraints
    let ineqs = A.into_iter().zip(b.into_iter()).map(|(A_i, b_i)| {
        let sense = ConstrSense::Less;
        let mut lhs = LinExpr::new();
        A_i.into_iter().zip(vars.iter()).for_each(|(A_ij, v)| {
            lhs.add_term(*A_ij, *v);
        });

        let lhs = Linear(lhs);
        let rhs = Constant(*b_i);
        IneqExpr { lhs, sense, rhs }
    });
    let _constrs: Vec<_> = ineqs
        .enumerate()
        .map(|(i, constr)| model.add_constr(&format!("r{}", i), constr))
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
