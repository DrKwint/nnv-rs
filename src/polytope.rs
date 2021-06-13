use crate::affine::Affine;
use crate::util::solve;
use good_lp::ResolutionError;
use ndarray::concatenate;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::ScalarOperand;
use ndarray::Zip;
use num::Float;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Polytope<T: Float> {
    halfspaces: Affine<T>,
    lower_bounds: Option<Array1<T>>,
    upper_bounds: Option<Array1<T>>,
}

impl<T: 'static + Float + Debug> Polytope<T>
where
    T: std::convert::Into<f64>,
    T: std::fmt::Display,
    T: ScalarOperand,
    f64: std::convert::From<T>,
{
    pub fn new(constraint_coeffs: Array2<T>, upper_bounds: Array1<T>) -> Self {
        Polytope {
            halfspaces: Affine::new(constraint_coeffs, upper_bounds),
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn from_affine(halfspaces: Affine<T>) -> Self {
        Polytope {
            halfspaces: halfspaces,
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn from_input_bounds(lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        Self {
            halfspaces: Affine::from_raw(Array2::zeros((lower_bounds.len() + 1, 1)), false),
            lower_bounds: Some(lower_bounds.clone()),
            upper_bounds: Some(upper_bounds.clone()),
        }
        //poly.with_input_bounds(lower_bounds, upper_bounds)
    }

    pub fn with_input_bounds(mut self, lower_bounds: Array1<T>, upper_bounds: Array1<T>) -> Self {
        self.lower_bounds = Some(lower_bounds.clone());
        self.upper_bounds = Some(upper_bounds.clone());
        let axis = if self.halfspaces.is_lhs {
            Axis(1)
        } else {
            Axis(0)
        };
        let lbs = concatenate(
            axis,
            &[
                (Array2::eye(lower_bounds.len()) * T::from(-1.).unwrap()).view(),
                lower_bounds.insert_axis(axis).view(),
            ],
        )
        .unwrap();
        self.add_constraints(&Affine::from_raw(lbs, self.halfspaces.is_lhs));
        let ubs = concatenate(
            axis,
            &[
                Array2::eye(upper_bounds.len()).view(),
                upper_bounds.insert_axis(axis).view(),
            ],
        )
        .unwrap();
        self.add_constraints(&Affine::from_raw(ubs, self.halfspaces.is_lhs));
        self
    }

    pub fn get_input_bounds(&self) -> Option<(ArrayView1<T>, ArrayView1<T>)> {
        self.lower_bounds
            .as_ref()
            .map(|x| x.view())
            .zip(self.upper_bounds.as_ref().map(|x| x.view()))
    }

    pub fn get_input_lower_bound(&self) -> Option<ArrayView1<T>> {
        self.lower_bounds.as_ref().map(|x| x.view())
    }

    pub fn get_input_upper_bound(&self) -> Option<ArrayView1<T>> {
        self.upper_bounds.as_ref().map(|x| x.view())
    }

    pub fn reduce_fixed_inputs(&self) -> Self {
        let lbs = self.lower_bounds.as_ref().unwrap();
        let ubs = self.upper_bounds.as_ref().unwrap();
        let is_fixed = Zip::from(lbs)
            .and(ubs)
            .map_collect(|&lb, &ub| if lb == ub { false } else { true });
        let fixed = Zip::from(lbs)
            .and(ubs)
            .map_collect(|&lb, &ub| if lb == ub { lb } else { T::zero() });
        let ub_reduction = if self.halfspaces.is_lhs {
            self.coeffs().dot(&fixed)
        } else {
            fixed.dot(&self.coeffs())
        };
        let new_eqn_ubs = &self.eqn_upper_bounds() - ub_reduction;
        let vars = self.get_coeffs_as_rows();
        let new_eqns_vec: Vec<ArrayView2<T>> = vars
            .columns()
            .into_iter()
            .zip(is_fixed)
            .filter(|(_, fixed)| *fixed)
            .map(|(var, _)| var.insert_axis(Axis(0)))
            .collect();
        let new_eqns = concatenate(Axis(0), new_eqns_vec.as_slice()).unwrap();

        Polytope {
            halfspaces: Affine::new(new_eqns, new_eqn_ubs),
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn coeffs(&self) -> ArrayView2<T> {
        self.halfspaces.get_mul()
    }

    pub fn get_coeffs_as_rows(&self) -> ArrayView2<T> {
        self.halfspaces.get_coeffs_as_rows()
    }

    pub fn eqn_upper_bounds(&self) -> ArrayView1<T> {
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
        Zip::from(self.eqn_upper_bounds())
            .and(&vals)
            .fold(true, |acc, ub, v| acc && (v <= ub))
    }

    /// Check whether the Star set is empty.
    pub fn is_empty(&self) -> bool {
        let mut c = Array1::zeros(self.eqn_upper_bounds().len());
        c[[0]] = T::one();

        let solved = solve(
            self.halfspaces.get_coeffs_as_rows().rows(),
            self.eqn_upper_bounds(),
            c.view(),
            self.lower_bounds.as_ref().map(|x| x.view()),
            self.upper_bounds.as_ref().map(|x| x.view()),
        )
        .0;
        match solved {
            Ok(_) | Err(ResolutionError::Unbounded) => false,
            _ => true,
        }
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
        let coeffs = array![[0., 0., 1.], [0., 0., -1.]];
        let ubs = array![0., -1.];
        let pt = Polytope::new(coeffs, ubs);
        println!("{:?}", pt);
        assert!(pt.is_empty());
    }

    #[test]
    fn test_member() {
        let coeffs = array![[1., 0.], [0., 1.], [1., -1.]].reversed_axes();
        let ubs = array![0., 0., 0.];
        let pt = Polytope::new(coeffs, ubs);
        println!("{:?}", pt);
        let _points = vec![array![0., 0.], array![1., 1.], array![0., 1.]];
    }
}
