use crate::affine::Affine;
use highs::ColProblem;
use highs::HighsModelStatus;
use highs::Row;
use highs::Sense;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Zip;
use num::Float;

#[derive(Clone, Debug)]
pub struct Polytope<T: Float> {
    halfspaces: Affine<T>,
}

impl<T: 'static + Float> Polytope<T>
where
    T: std::convert::Into<f64>,
    T: std::fmt::Display,
{
    pub fn new(constraint_coeffs: Array2<T>, upper_bounds: Array1<T>) -> Self {
        Polytope {
            halfspaces: Affine::new(constraint_coeffs, upper_bounds),
        }
    }

    pub fn from_affine(halfspaces: Affine<T>) -> Self {
        Polytope {
            halfspaces: halfspaces,
        }
    }

    pub fn coeffs(&self) -> ArrayView2<T> {
        self.halfspaces.get_mul()
    }

    pub fn upper_bounds(&self) -> ArrayView1<T> {
        self.halfspaces.get_shift()
    }

    fn num_dims(&self) -> usize {
        self.halfspaces.input_dim()
    }

    pub fn num_constraints(&self) -> usize {
        self.halfspaces.output_dim()
    }

    pub fn add_constraints(&mut self, constraints: &Affine<T>) {
        self.halfspaces.add_eqns(constraints);
    }

    pub fn affine_map(&self, affine: &Affine<T>) -> Affine<T> {
        assert_eq!(self.halfspaces.is_lhs, affine.is_lhs);
        if self.halfspaces.is_lhs {
            self.halfspaces.lhs_mul(affine)
        } else {
            self.halfspaces.rhs_mul(affine)
        }
    }

    pub fn is_member(&self, point: &ArrayView1<T>) -> bool {
        let vals = point.dot(&self.coeffs());
        Zip::from(self.upper_bounds())
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
    }

    /// Check whether the Star set is empty.
    pub fn is_empty(&self) -> bool {
        let coeffs = self.halfspaces.get_mul();
        let upper_bounds = self.halfspaces.get_shift();
        let mut pb = ColProblem::default();

        let problem_cs = upper_bounds.mapv(|x| pb.add_row(..=x)).to_vec();
        let eqn_iter = if self.halfspaces.is_lhs {
            coeffs.columns().into_iter()
        } else {
            coeffs.columns().into_iter()
        };
        eqn_iter.for_each(|var| {
            let x: Vec<(Row, f64)> = problem_cs
                .clone()
                .into_iter()
                .zip(var.to_vec().into_iter().map(|x| T::into(x)))
                .collect();
            pb.add_column(1., (0.).., x);
        });
        let mut model = pb.optimise(Sense::Maximise);
        model.make_quiet();
        let solved = model.solve();
        let status = solved.status();
        !((status == HighsModelStatus::Optimal) || (status == HighsModelStatus::PrimalUnbounded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_empty() {
        let coeffs = array![[0., 1.], [1., 0.]];
        let ubs = array![0., 0.];
        let pt = Polytope::new(coeffs, ubs);
        assert!(!pt.is_empty());
        let coeffs = array![[0., 1.], [1., 0.], [0., -1.]];
        let ubs = array![0., 0., -1.];
        let pt = Polytope::new(coeffs, ubs);
        assert!(pt.is_empty());
    }

    #[test]
    fn test_member() {
        let coeffs = array![[1., 0.], [0., 1.], [1., -1.]].reversed_axes();
        let ubs = array![0., 0., 0.];
        let pt = Polytope::new(coeffs, ubs);
        println!("{:?}", pt);
        let points = vec![array![0., 0.], array![1., 1.], array![0., 1.]];
    }
}
