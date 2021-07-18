//! Data structure representing the paths through a DNN with sets as input/output
use crate::affine::Affine;
use crate::star::Star;
use crate::DNN;
use indextree::{Arena, NodeId};
use ndarray::Dimension;
use ndarray::{Array1, Array2};
use ndarray::{Ix2, Ix4};
use num::Float;
use rand::distributions::{Bernoulli, Distribution};

/// Data structure representing the paths through a deep neural network (DNN)
pub struct Constellation<T: Float> {
    arena: Arena<StarNode<T>>,
    dnn: DNN<T>,
}

impl<T: Float> Constellation<T>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::ops::MulAssign
        + std::fmt::Display
        + std::fmt::Debug,
    f64: std::convert::From<T>,
{
    /// Instantiate a Constellation with given input set and network
    pub fn new(input_star: PolyStar<T>, dnn: DNN<T>) -> Self {
        let star_node = StarNode {
            star: input_star,
            dnn_layer: 0,
            remaining_steps: 0,
            cdf: None,
            output_bounds: None,
            is_expanded: false,
        };
        let mut arena = Arena::new();
        let root = arena.new_node(star_node);
        Self {
            arena: arena,
            dnn: dnn,
        }
    }

    fn expand_node(&mut self, id: NodeId) -> Vec<NodeId> {
        let node = self.arena.get_mut(id).unwrap().get_mut();
        let children = node.expand(&self.dnn);
        children
            .into_iter()
            .map(|x| self.arena.new_node(x))
            .collect()
    }

    /*
    pub fn overestimate_output_range(&self, star: Star<T>, layer: usize) -> (T, T) {
        let est_star = self.dnn_affines[layer..]
            .iter()
            .fold(star, |x: Star<T>, a| x.affine_map(a));
        (est_star.get_min(0), est_star.get_max(0))
    }

    /// Sample from a Gaussian distribution with an upper bound on the output value
    ///
    /// This function uses a two step sampling process. The first walks the Constellation's binary
    /// tree, which is defined by the parameters of the underlying neural network. This walk
    /// continues until a safe star has been reached, and then it's sampled in the second step to
    /// produce the final set of samples.
    ///
    /// # Arguments
    ///
    /// * `loc` - Gaussian location parameter
    /// * `scale` - Gaussian covariance matrix or, if diagonal, vector
    /// * `safe_value` - Maximum value an output can have to be considered safe
    /// * `cdf_samples` - Number of samples to use in estimating a polytope's CDF under the Gaussian
    /// * `num_samples` - Number of samples to return from the final star that's sampled
    ///
    /// # Panics
    ///
    /// # Returns
    ///
    /// (Vec of samples, Array of sample probabilities, branch probability)
    #[allow(clippy::too_many_lines)]
    pub fn sample_multivariate_gaussian(
        &mut self,
        loc: &Array1<T>,
        scale: &Array2<T>,
        safe_value: T,
        cdf_samples: usize,
        num_samples: usize,
        max_iters: usize,
    ) -> (Vec<(Array1<f64>, f64)>, f64) {
        let mut current_node = 0;
        let mut rng = rand::thread_rng();
        let mut path = vec![];
        let mut path_logp = 0.;
        let output: Vec<(Array1<f64>, f64)>;
        loop {
            if current_node == 0 {
                path = vec![0];
                path_logp = 0.
            } else {
                path.push(current_node);
            }
            // base case for feasability
            if current_node == 0 && !self.arena[current_node].is_feasible {
                output = self.arena[current_node].star.trunc_gaussian_sample(
                    loc,
                    scale,
                    num_samples,
                    max_iters,
                );
                break;
            }
            let bounds = self.overestimate_output_range(
                self.arena[current_node].star.clone(),
                self.arena[current_node].layer,
            );
            if self.arena[current_node].cdf.is_none() {
                self.arena[current_node].cdf = Some(
                    self.arena[current_node]
                        .star
                        .trunc_gaussian_cdf(loc, scale, cdf_samples, max_iters)
                        .0,
                );
            }
            current_node = if bounds.1 <= safe_value {
                // sample current node
                output = self.arena[current_node].star.trunc_gaussian_sample(
                    loc,
                    scale,
                    num_samples,
                    max_iters,
                );
                break;
            } else if bounds.0 > safe_value {
                // at root, if lb is too high then no output will be safe, so just sample
                if current_node == 0 {
                    output = self.arena[current_node].star.trunc_gaussian_sample(
                        loc,
                        scale,
                        num_samples,
                        max_iters,
                    );
                    break;
                }
                // restart
                self.arena[current_node].is_feasible = false;
                let current_cdf = self.arena[current_node].cdf.unwrap();
                for idx in &path {
                    if let Some(x) = self.arena[*idx].cdf.as_mut() {
                        *x -= current_cdf
                    };
                }
                0
            } else {
                // expand
                let children = self.expand_node(current_node);
                match children.len() {
                    // leaf node
                    0 => {
                        output = self.arena[current_node].star.trunc_gaussian_sample(
                            loc,
                            scale,
                            num_samples,
                            max_iters,
                        );
                        break;
                    }
                    // affine or pos def stepRelu
                    1 => {
                        if self.arena[children[0]].is_feasible {
                            children[0]
                        } else {
                            self.arena[current_node].is_feasible = false;
                            let current_cdf = self.arena[current_node].cdf.unwrap();
                            for idx in &path {
                                if let Some(x) = self.arena[*idx].cdf.as_mut() {
                                    *x -= current_cdf
                                }
                            }
                            0
                        }
                    }
                    // stepRelu with pos and neg children
                    // In this branch, there can only be exactly two children
                    _ => {
                        match (
                            self.arena[children[0]].is_feasible,
                            self.arena[children[1]].is_feasible,
                        ) {
                            (false, false) => {
                                self.arena[current_node].is_feasible = false;
                                let current_cdf = self.arena[current_node].cdf.unwrap();
                                for idx in &path {
                                    if let Some(x) = self.arena[*idx].cdf.as_mut() {
                                        *x -= current_cdf
                                    };
                                }
                                0
                            }
                            (false, true) => children[1],
                            (true, false) => children[0],
                            (true, true) => {
                                // Bernoulli trial to choose child
                                let a_prob = if self.arena[children[0]].cdf.is_some() {
                                    self.arena[children[0]].cdf.unwrap()
                                } else {
                                    let prob = self.arena[children[0]]
                                        .star
                                        .trunc_gaussian_cdf(loc, scale, cdf_samples, max_iters)
                                        .0;
                                    self.arena[children[0]].cdf = Some(prob);
                                    prob
                                };
                                let b_prob = if self.arena[children[1]].cdf.is_some() {
                                    self.arena[children[0]].cdf.unwrap()
                                } else {
                                    let prob = self.arena[children[1]]
                                        .star
                                        .trunc_gaussian_cdf(loc, scale, cdf_samples, max_iters)
                                        .0;
                                    self.arena[children[0]].cdf = Some(prob);
                                    prob
                                };
                                match Bernoulli::new(a_prob / (a_prob + b_prob)) {
                                    // bernoulli error restart
                                    Err(_) => {
                                        self.arena[current_node].is_feasible = false;
                                        let current_cdf = self.arena[current_node].cdf.unwrap();
                                        for idx in &path {
                                            if let Some(x) = self.arena[*idx].cdf.as_mut() {
                                                *x -= current_cdf
                                            };
                                        }
                                        0
                                    } // if there's an error, return to the root
                                    Ok(bernoulli) => {
                                        if bernoulli.sample(&mut rng) {
                                            path_logp -= (a_prob / (a_prob + b_prob)).ln();
                                            children[0]
                                        } else {
                                            path_logp -= (b_prob / (a_prob + b_prob)).ln();
                                            children[1]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            };
        }
        (output, path_logp)
    }

    fn add_node(
        &mut self,
        star: Star<T>,
        parent: usize,
        layer: usize,
        remaining_steps: Option<usize>,
    ) -> usize {
        let idx = self.arena.len();
        let node = StarNode::new(idx, Some(parent), star, layer, remaining_steps);
        self.arena.push(node);
        idx
    }
    */
}

#[derive(Debug, Clone)]
/// Convenient uniform handling of different star types
pub enum PolyStar<T: Float> {
    VecStar(Star<T, Ix2>),
    ImgStar(Star<T, Ix4>),
}

impl<T: Float> PolyStar<T>
where
    T: std::convert::From<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    fn step_relu(&self, idx: usize) -> Vec<PolyStar<T>> {
        let stars = match self {
            PolyStar::VecStar(star) => star.step_relu(idx),
            PolyStar::ImgStar(star) => panic!(), //star.step_relu(idx),
        };
        todo!();
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct StarNode<T: num::Float> {
    star: PolyStar<T>,
    dnn_layer: usize,
    remaining_steps: usize,
    cdf: Option<f64>,
    output_bounds: Option<(f64, f64)>,
    is_expanded: bool,
}

impl<T: num::Float> StarNode<T>
where
    T: std::convert::From<f64>
        + std::convert::Into<f64>
        + ndarray::ScalarOperand
        + std::fmt::Display
        + std::fmt::Debug
        + std::ops::MulAssign,
    f64: std::convert::From<T>,
{
    pub fn expand(&self, dnn: &DNN<T>) -> Vec<Self> {
        // check if there is a step relu to do
        if self.remaining_steps > 0 {
            let new_child_stars = self.star.step_relu(self.remaining_steps);
            let new_remaining_steps = self.remaining_steps - 1;
            new_child_stars
                .into_iter()
                .map(|star| Self {
                    star,
                    dnn_layer: self.dnn_layer,
                    remaining_steps: new_remaining_steps,
                    cdf: None,
                    output_bounds: None,
                    is_expanded: false,
                })
                .collect()
        } else {
            if let Some(layer) = dnn.get_layer(self.dnn_layer + 1) {
                layer.apply(&self)
            } else {
                // leaf node
                vec![]
            }
        }
    }
}

/*
#[cfg(test)]
mod tests {
    extern crate ndarray_rand;
    use super::*;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

    #[test]
    fn it_works() {
        let star = Star::default(2);
        let normal = Normal::new(0., 1.).unwrap();
        let layers = vec![(3, 2), (2, 1)];
        let a = layers.iter().map(|&x| Array2::random(x, normal));
        let b = layers.iter().map(|&x| Array1::random(x.1, normal));
        let dnn = a.zip(b).map(|x| Affine::new(x.0, x.1)).collect();

        let _constellation = Constellation::new(star, dnn);
    }

    #[test]
    fn sample_test() {
        let dist = Normal::new(0., 1.).unwrap();
        let generate_layer = |in_, out_| {
            Affine::new(
                Array2::random((in_, out_), dist),
                Array1::random(out_, dist),
            )
        };
        let star = Star::default(4);
        let a = generate_layer(4, 64);
        let b = generate_layer(64, 2);
        let c = generate_layer(2, 1);
        let mut constellation = Constellation::new(star, vec![a, b, c]);

        let loc = Array1::zeros(4);
        let scale = Array2::eye(4);
        let val = constellation.sample_multivariate_gaussian(&loc, &scale, -100., 10000, 10, 10);
        print!("{:?}", val);
    }
}
*/
