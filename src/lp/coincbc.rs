#![allow(non_snake_case)]
use crate::bounds::Bounds1;
use crate::lp::LinearSolution;
use crate::NNVFloat;
use good_lp::solvers::coin_cbc::coin_cbc;
use good_lp::Solution;
use good_lp::{variable, ResolutionError, SolverModel};
use good_lp::{Expression, IntoAffineExpression, ProblemVariables, Variable};
use ndarray::{Array1, ArrayView1};

/// An linear expression without a constant component
#[derive(Clone)]
pub struct LinearExpression {
    pub coefficients: Vec<(Variable, f64)>,
}

impl IntoAffineExpression for LinearExpression {
    type Iter = std::vec::IntoIter<(Variable, f64)>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.coefficients.into_iter()
    }
}

/// Minimizes the expression `c` given the constraint `Ax < b`.
/// # Panics
pub fn solve<'a, I, J>(
    A: I,
    b: J,
    var_coeffs: ArrayView1<NNVFloat>,
    var_bounds_opt: Option<&Bounds1>,
) -> LinearSolution
where
    I: IntoIterator<Item = ArrayView1<'a, NNVFloat>>,
    J: IntoIterator<Item = &'a NNVFloat>,
{
    let mut _shh_out;
    let mut _shh_err;
    if !cfg!(test) {
        _shh_out = shh::stdout().unwrap();
        _shh_err = shh::stderr().unwrap();
    }
    let mut problem = ProblemVariables::new();
    let vars: Vec<_> = if let Some(bounds) = var_bounds_opt {
        bounds
            .bounds_iter()
            .into_iter()
            .map(|b| problem.add(variable().bounds(b[[0]]..b[[1]])))
            .collect()
    } else {
        var_coeffs.iter().map(|_| problem.add(variable())).collect()
    };
    let c_expression = LinearExpression {
        coefficients: vars
            .iter()
            .copied()
            .zip(var_coeffs.iter().copied())
            .collect(),
    };
    let mut unsolved = problem.minimise(c_expression.clone()).using(coin_cbc);

    A.into_iter()
        .zip(b.into_iter())
        .for_each(|pair: (ArrayView1<NNVFloat>, &NNVFloat)| {
            let (coeffs, ub) = pair;
            let expr = LinearExpression {
                coefficients: vars.iter().copied().zip(coeffs.iter().copied()).collect(),
            };
            let constr = good_lp::constraint::leq(Expression::from_other_affine(expr), *ub);
            unsolved.add_constraint(constr);
        });

    let raw_soln_result = unsolved.solve();
    match raw_soln_result {
        Ok(raw_soln) => {
            let cbc_model = raw_soln.model();
            match cbc_model.secondary_status() {
                coin_cbc::raw::SecondaryStatus::HasSolution => {
                    let param = Array1::from_iter(cbc_model.col_solution().iter().copied());
                    let fun = raw_soln.eval(c_expression);
                    LinearSolution::Solution(param, fun)
                }
                _ => todo!(),
            }
        }
        Err(ResolutionError::Infeasible | ResolutionError::Other(_)) => LinearSolution::Infeasible,
        Err(ResolutionError::Unbounded) => LinearSolution::Unbounded(Array1::zeros(1)),
        Err(e) => panic!("{:?}", e),
    }
}
