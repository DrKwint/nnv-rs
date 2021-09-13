use crate::affine::Affine2;
use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly;
use crate::dnn::DNNIndex;
use crate::dnn::DNNIterator;
use crate::dnn::DNN;
use crate::star::Star;
use log::debug;
use log::trace;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use num::Float;
use rand::Rng;
use std::fmt::Debug;
use std::iter::Sum;

#[derive(Debug, Clone)]
pub enum StarNodeOp<T: num::Float> {
    Leaf,
    Affine(Affine2<T>),
    StepRelu(usize),
    StepReluDropout((T, usize)),
}

impl<T: num::Float + Default + Sum + Debug + 'static> StarNodeOp<T> {
    /// bounds: Output bounds: Concrete bounds
    /// affs: Input bounds: Abstract bounds in terms of inputs
    pub fn apply_bounds(
        &self,
        bounds: &Bounds1<T>,
        lower_aff: &Affine2<T>,
        upper_aff: &Affine2<T>,
    ) -> (Bounds1<T>, (Affine2<T>, Affine2<T>)) {
        match self {
            Self::Leaf => (bounds.clone(), (lower_aff.clone(), upper_aff.clone())),
            Self::Affine(aff) => {
                println!("Aff {:?}", &aff);
                (
                    aff.signed_apply(&bounds),
                    (
                        aff.signed_compose(&lower_aff, &upper_aff),
                        aff.signed_compose(&upper_aff, &lower_aff),
                    ),
                )
            }
            Self::StepRelu(dim) => crate::deeppoly::deep_poly_steprelu(
                *dim,
                bounds.clone(),
                lower_aff.clone(),
                upper_aff.clone(),
            ),
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StarNodeType<T: num::Float> {
    Leaf,
    Affine {
        child_idx: usize,
    },
    StepRelu {
        dim: usize,
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
    },
    StepReluDropOut {
        dim: usize,
        dropout_prob: T,
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
        trd_child_idx: Option<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct StarNode<T: num::Float, D: Dimension> {
    star: Star<T, D>,
    dnn_index: DNNIndex,
    star_cdf: Option<T>,
    output_bounds: Option<(T, T)>,
    is_feasible: bool,
}

impl<T: num::Float, D: Dimension> StarNode<T, D> {
    pub fn default(star: Star<T, D>) -> Self {
        Self {
            star,
            dnn_index: DNNIndex::default(),
            star_cdf: None,
            output_bounds: None,
            is_feasible: true,
        }
    }

    pub fn with_dnn_index(mut self, dnn_index: DNNIndex) -> Self {
        self.dnn_index = dnn_index;
        self
    }
}

impl<T: num::Float, D: Dimension> StarNode<T, D>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign,
    f64: std::convert::From<T>,
{
    pub fn get_star(&self) -> &Star<T, D> {
        &self.star
    }

    pub fn get_dnn_index(&self) -> DNNIndex {
        self.dnn_index
    }

    pub fn get_feasible(&self) -> bool {
        self.is_feasible
    }

    pub fn set_feasible(&mut self, val: bool) {
        self.is_feasible = val
    }

    pub fn get_expanded(&self) -> bool {
        todo!()
    }

    pub fn gaussian_cdf(
        &mut self,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
    ) -> T {
        self.star_cdf.map_or_else(
            || {
                let out = self.star.trunc_gaussian_cdf(mu, sigma, n, max_iters);
                let cdf = out.0.into();
                self.star_cdf = Some(cdf);
                cdf
            },
            |cdf| cdf,
        )
    }

    pub fn set_cdf(&mut self, val: T) {
        self.star_cdf = Some(val);
    }

    /// # Panics
    pub fn add_cdf(&mut self, add: T) {
        if let Some(ref mut cdf) = self.star_cdf {
            *cdf += add
        } else {
            // TODO
        }
    }
}

impl<T: Float> StarNode<T, Ix2>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::default::Default
        + std::iter::Sum
        + approx::AbsDiffEq,
    f64: std::convert::From<T>,
{
    pub fn gaussian_sample<R: Rng>(
        &self,
        rng: &mut R,
        mu: &Array1<T>,
        sigma: &Array2<T>,
        n: usize,
        max_iters: usize,
        input_bounds: &Option<Bounds1<T>>,
    ) -> Vec<(Array1<T>, T)> {
        self.star
            .clone()
            .gaussian_sample(rng, mu, sigma, n, max_iters, input_bounds)
            .iter()
            .map(|(arr, val)| (arr.mapv(|x| x.into()), num::NumCast::from(*val).unwrap()))
            .collect()
    }

    pub fn get_output_bounds(
        &mut self,
        dnn: &DNN<T>,
        output_fn: &dyn Fn(Bounds1<T>) -> (T, T),
    ) -> (T, T) {
        self.output_bounds.map_or_else(
            || {
                trace!("get_output_bounds on star {:?}", self.star);
                let bounding_box = self.star.calculate_axis_aligned_bounding_box();
                // TODO: update this to use DeepPoly to get proper bounds rather than this estimate
                /*
                let bounds = {
                    let out_star = dnn
                        .get_layers()
                        .iter()
                        .skip(self.dnn_layer)
                        .fold(self.star.clone(), |s: Star<T, Ix2>, l: &Layer<T>| {
                            l.apply_star2(s)
                        });
                    (out_star.clone().get_min(idx), out_star.get_max(idx))
                };
                self.output_bounds = Some(bounds);
                */
                output_fn(deep_poly(
                    bounding_box,
                    DNNIterator::new(dnn, self.dnn_index),
                ))
            },
            |bounds| bounds,
        )
    }
}
