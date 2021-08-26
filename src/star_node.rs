use crate::bounds::Bounds1;
use crate::deeppoly::deep_poly;
use crate::dnn::DNNIndex;
use crate::dnn::DNNIterator;
use crate::dnn::DNN;
use crate::star::Star;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{Array1, Array2};
use num::Float;
use rand::Rng;

#[derive(Debug, Clone)]
pub enum StarNodeType {
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
        fst_child_idx: usize,
        snd_child_idx: Option<usize>,
        trd_child_idx: Option<usize>,
    },
}

#[derive(Debug)]
pub struct StarNode<T: num::Float, D: Dimension> {
    star: Star<T, D>,
    children: Option<StarNodeType>,
    dnn_index: DNNIndex,
    star_cdf: Option<T>,
    output_bounds: Option<(T, T)>,
    is_feasible: bool,
}

impl<T: num::Float, D: Dimension> StarNode<T, D> {
    pub fn default(star: Star<T, D>) -> Self {
        Self {
            star,
            children: None,
            dnn_index: DNNIndex::default(),
            star_cdf: None,
            output_bounds: None,
            is_feasible: true,
        }
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
            todo!()
        }
    }

    pub fn get_child_ids(&self) -> Option<Vec<usize>> {
        match self.children {
            Some(StarNodeType::Leaf) => vec![],
            Some(StarNodeType::Affine { child_idx }) => vec![child_idx],
            Some(StarNodeType::StepRelu {
                dim,
                fst_child_idx,
                snd_child_idx,
            }) => {
                let child_ids: Vec<usize> = Vec::new();
                child_ids.push(fst_child_idx);
                if let Some(idx) = snd_child_idx {
                    child_ids.push(idx);
                }
                child_ids
            }
            Some(StarNodeType::StepReluDropOut {
                dim: usize,
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
            }) => {
                let child_ids: Vec<usize> = Vec::new();
                child_ids.push(fst_child_idx);
                if let Some(idx) = snd_child_idx {
                    child_ids.push(idx);
                }
                if let Some(idx) = trd_child_idx {
                    child_ids.push(idx);
                }
                child_ids
            }
            None => None,
            _ => todo!(),
        }
    }
}

/// Expand a node's children, inserting them into the arena.
///
/// # Arguments
///
/// * `self` - The node to expand
/// * `node_arena` - The data structure storing star nodes
/// * `dnn_iter` - The iterator of operations in the dnn
///
/// # Returns
/// * `child_ids` - A vector containing the ids of the child nodes
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
        + std::iter::Sum,
    f64: std::convert::From<T>,
{
    pub fn expand(
        &mut self,
        node_arena: &mut Vec<Self>,
        dnn_iter: &mut DNNIterator<T>,
    ) -> Vec<usize> {
        if self.children.is_some() {
            self.get_child_ids()
        } else {
            // Get this node's operation from the dnn_iter
            self.children = dnn_iter.next();
            // Do this node's operation to produce its children
            match self.children {
                Some(StarNodeType::Leaf) => vec![],
                Some(StarNodeType::StepRelu {
                    dim,
                    fst_child_idx,
                    snd_child_idx,
                }) => {
                    let child_stars = self.star.clone().step_relu2(dim);
                    child_stars
                        .into_iter()
                        .map(|star| {
                            let idx = node_arena.len();
                            node_arena.push(Self {
                                star,
                                children: None,
                                dnn_index: dnn_iter.get_idx(),
                                star_cdf: None,
                                output_bounds: None,
                                is_feasible: false,
                            });
                            idx
                        })
                        .collect()
                }
                None => panic!(),
                _ => todo!(),
            }
        }
    }

    /*
        /// # Panics
        pub fn expand_old(&self, dnn: &DNN<T>) -> Vec<Self> {
            // check if there is a step relu to do
            if let Some(relu_step) = self.remaining_steps {
                let new_child_stars = self.star.clone().step_relu2(relu_step);
                let new_remaining_steps = if relu_step == 0 {
                    None
                } else {
                    Some(relu_step - 1)
                };
                new_child_stars
                    .into_iter()
                    .map(|star| Self {
                        star,
                        dnn_layer: self.dnn_layer,
                        remaining_steps: new_remaining_steps,
                        star_cdf: None,
                        output_bounds: None,
                        is_expanded: false,
                        is_feasible: true,
                    })
                    .collect()
            } else if let Some(layer) = dnn.get_layer(self.dnn_layer) {
                vec![Self {
                    star: layer.apply_star2(self.star.clone()),
                    dnn_layer: self.dnn_layer + 1,
                    remaining_steps: Some(
                        dnn.get_layer(self.dnn_layer).unwrap().output_shape()[-1].unwrap() - 1,
                    ),
                    star_cdf: None,
                    output_bounds: None,
                    is_expanded: false,
                    is_feasible: true,
                }]
            } else {
                // leaf node
                vec![]
            }
        }
    */

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
        idx: usize,
        output_fn: &dyn Fn(Bounds1<T>) -> (T, T),
    ) -> (T, T) {
        self.output_bounds.map_or_else(
            || {
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
                output_fn(deep_poly(bounding_box, dnn))
            },
            |bounds| bounds,
        )
    }
}

trait ArenaLike<T> {
    fn new_node(&mut self, data: T) -> usize;
}

impl<T: num::Float, D: Dimension> ArenaLike<StarNode<T, D>> for Vec<StarNode<T, D>> {
    fn new_node(&mut self, data: StarNode<T, D>) -> usize {
        let new_id = self.len();
        self.push(data);
        new_id
    }
}
