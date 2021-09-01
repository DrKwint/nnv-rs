use crate::affine::Affine2;
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
pub enum StarNodeOp<T: num::Float> {
    Leaf,
    Affine(Affine2<T>),
    StepRelu(usize),
    StepReluDropout((T, usize)),
}

impl<T: num::Float> StarNodeOp<T> {}

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
    children: Option<StarNodeType<T>>,
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

    pub fn get_index(&self) -> DNNIndex {
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
            todo!()
        }
    }

    pub fn get_child_ids(&self) -> Option<Vec<usize>> {
        match self.children {
            Some(StarNodeType::Leaf) => Some(vec![]),
            Some(StarNodeType::Affine { child_idx }) => Some(vec![child_idx]),
            Some(StarNodeType::StepRelu {
                dim,
                fst_child_idx,
                snd_child_idx,
            }) => {
                let mut child_ids: Vec<usize> = Vec::new();
                child_ids.push(fst_child_idx);
                if let Some(idx) = snd_child_idx {
                    child_ids.push(idx);
                }
                Some(child_ids)
            }
            Some(StarNodeType::StepReluDropOut {
                dim: usize,
                dropout_prob: T,
                fst_child_idx,
                snd_child_idx,
                trd_child_idx,
            }) => {
                let mut child_ids: Vec<usize> = Vec::new();
                child_ids.push(fst_child_idx);
                if let Some(idx) = snd_child_idx {
                    child_ids.push(idx);
                }
                if let Some(idx) = trd_child_idx {
                    child_ids.push(idx);
                }
                Some(child_ids)
            }
            None => None,
            _ => todo!(),
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
        + std::iter::Sum,
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

    /// Expand a node's children, possibly inserting them into the arena.
    ///
    /// # Arguments
    ///
    /// * `self` - The node to expand
    /// * `node_arena` - The data structure storing star nodes
    /// * `dnn_iter` - The iterator of operations in the dnn
    ///
    /// # Returns
    /// * `child_ids` - A vector containing the ids of the child nodes
    pub fn get_children(
        &mut self,
        arena: &mut Vec<StarNode<T, Ix2>>,
        dnn: &DNN<T>,
    ) -> StarNodeType<T> {
        if let Some(children) = self.children {
            return children;
        }

        let dnn_iter = &mut DNNIterator::new(dnn, self.dnn_index);

        // Get this node's operation from the dnn_iter
        let op = dnn_iter.next();
        // Do this node's operation to produce its children
        match op {
            Some(StarNodeOp::Leaf) => {
                self.children = Some(StarNodeType::Leaf);
            }
            Some(StarNodeOp::Affine(aff)) => {
                let idx = arena.len();
                let child_idx = arena.new_node(
                    StarNode::default(self.star.clone().affine_map2(&aff))
                        .with_dnn_index(dnn_iter.get_idx()),
                );
                self.children = Some(StarNodeType::Affine { child_idx });
            }
            Some(StarNodeOp::StepRelu(dim)) => {
                let child_stars = self.star.clone().step_relu2(dim);
                let ids: Vec<usize> = child_stars
                    .into_iter()
                    .map(|star| {
                        let idx = arena.len();
                        arena.push(StarNode::default(star).with_dnn_index(dnn_iter.get_idx()));
                        idx
                    })
                    .collect();
                self.children = Some(StarNodeType::StepRelu {
                    dim,
                    fst_child_idx: ids[0],
                    snd_child_idx: ids.get(1).cloned(),
                });
            }
            Some(StarNodeOp::StepReluDropout((dropout_prob, dim))) => {
                let child_stars = self.star.clone().step_relu2_dropout(dim);
                let ids: Vec<usize> = child_stars
                    .into_iter()
                    .map(|star| {
                        let idx = arena.len();
                        arena.push(StarNode::default(star).with_dnn_index(dnn_iter.get_idx()));
                        idx
                    })
                    .collect();
                self.children = Some(StarNodeType::StepReluDropOut {
                    dim,
                    dropout_prob,
                    fst_child_idx: ids[0],
                    snd_child_idx: ids.get(1).cloned(),
                    trd_child_idx: ids.get(2).cloned(),
                });
            }
            None => panic!(),
            _ => todo!(),
        };
        self.children.unwrap()
    }
}

pub trait ArenaLike<T> {
    fn new_node(&mut self, data: T) -> usize;
}

impl<T: num::Float, D: Dimension> ArenaLike<StarNode<T, D>> for Vec<StarNode<T, D>> {
    fn new_node(&mut self, data: StarNode<T, D>) -> usize {
        let new_id = self.len();
        self.push(data);
        new_id
    }
}
