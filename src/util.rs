extern crate good_lp;

use good_lp::solvers::highs::highs;
use good_lp::solvers::highs::HighsSolution;
use good_lp::Expression;
use good_lp::IntoAffineExpression;
use good_lp::ProblemVariables;
use good_lp::ResolutionError;
use good_lp::Variable;
use good_lp::{variable, Solution, SolverModel};
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Axis;
use ndarray::Slice;
use ndarray_linalg::EigVals;
use ndarray_linalg::SVD;
use ndarray::s;
use std::collections::HashMap;

pub fn pinv(x: &Array2<f64>) -> Array2<f64> {
    let (u_opt, sigma, vt_opt) = x.svd(true, true).unwrap();
    let u = u_opt.unwrap();
    let vt = vt_opt.unwrap();
    let sig_diag = &sigma.map(|x| if *x < 1e-10 { 0. } else { 1. / x });
    let mut sig_base = Array2::eye(u.nrows());
    sig_base.diag_mut().slice_mut(s![..sig_diag.len()]).assign(sig_diag);
    let sig = sig_base.slice_axis(Axis(0), Slice::from(..vt.nrows()));
    vt.t().dot(&sig.dot(&u.t()))
}

pub fn ensure_spd(A: Array2<f64>) -> Array2<f64> {
    let B = (&A + &A.t()) / 2.;
    let (_, sigma, vt_opt) = A.svd(false, true).unwrap();
    let vt = vt_opt.unwrap();
    let H = vt.t().dot(&sigma).dot(&vt);
    let mut a_hat = (B + H) / 2.;
    // ensure symmetry
    a_hat = (&a_hat + &a_hat.t()) / 2.;
    let min_eig = a_hat.eigvals().unwrap();
    println!("min_eig {}", min_eig);
    a_hat
}

/// An linear expression without a constant component
#[derive(Clone)]
struct LinearExpression {
    coefficients: HashMap<Variable, f64>,
}

impl IntoAffineExpression for LinearExpression {
    type Iter = std::collections::hash_map::IntoIter<Variable, f64>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.coefficients.into_iter()
    }
}

pub fn solve<'a, I, T: 'a>(
    A: I,
    b: ArrayView1<T>,
    c: ArrayView1<T>,
    c_lower_bounds: Option<ArrayView1<T>>,
    c_upper_bounds: Option<ArrayView1<T>>,
) -> (Result<HighsSolution, ResolutionError>, Option<f64>)
where
    T: std::convert::Into<f64> + std::clone::Clone + std::marker::Copy,
    I: IntoIterator<Item = ArrayView1<'a, T>>,
    f64: std::convert::From<T>,
{
    let shh = shh::stdout().unwrap();
    let mut problem = ProblemVariables::new();
    let vars = if c_lower_bounds.is_some() {
        let lowers = c_lower_bounds.unwrap();
        let uppers = c_upper_bounds.unwrap();
        lowers
            .into_iter()
            .zip(uppers)
            .map(|bounds| {
                problem.add(
                    variable()
                        .min(f64::from(*bounds.0))
                        .max(f64::from(*bounds.1)),
                )
            })
            .collect()
    } else {
        problem.add_vector(variable(), c.len())
    };
    let c_expression = LinearExpression {
        coefficients: vars
            .iter()
            .cloned()
            .zip(c.iter().map(|x| f64::from(*x)))
            .collect(),
    };
    let mut unsolved = problem.minimise(c_expression.clone()).using(highs);

    A.into_iter()
        .zip(b.into_iter())
        .for_each(|pair: (ArrayView1<T>, &T)| {
            let (coeffs, ub) = pair;
            let expr = LinearExpression {
                coefficients: vars
                    .iter()
                    .cloned()
                    .zip(coeffs.iter().map(|x| f64::from(*x)))
                    .collect(),
            };
            let constr =
                good_lp::constraint::leq(Expression::from_other_affine(expr), f64::from(*ub));
            unsolved.add_constraint(constr);
        });
    let soln = unsolved.solve();
    let fun = soln.as_ref().ok().map(|x| x.eval(c_expression));
    (soln, fun)
}

/*
pub fn solve<'a, I, T: 'a>(A: I, b: ArrayView1<T>, c: ArrayView1<T>) -> SolvedModel
where
    T: std::convert::Into<f64> + std::clone::Clone + std::marker::Copy,
    I: IntoIterator<Item = ArrayView1<'a, T>>,
{
    let mut pb = ColProblem::default();

    let problem_cs = b.mapv(|x| pb.add_row(..=x)).to_vec();
    A.into_iter().for_each(|var: ArrayView1<T>| {
        let x: Vec<(Row, f64)> = problem_cs
            .clone()
            .into_iter()
            .zip(var.into_iter().map(|x| T::into(*x)))
            .collect();
        pb.add_column(1., (0.).., x);
    });
    let mut model = pb.optimise(Sense::Maximise);
    model.make_quiet();
    model.solve()
}
*/
