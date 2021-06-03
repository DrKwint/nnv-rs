extern crate rand;

use crate::affine::Affine;
use crate::star::Star;
use ndarray::Array1;
use ndarray::Array2;
use num::Float;
use rand::distributions::{Bernoulli, Distribution};

#[derive(Debug)]
pub struct Constellation<T: Float> {
    arena: Vec<StarNode<T>>,
    dnn_affines: Vec<Affine<T>>,
}

impl<T: Float> Constellation<T>
where
    T: std::convert::From<f64>,
    T: std::convert::Into<f64>,
    T: ndarray::ScalarOperand,
    T: std::fmt::Display,
    T: std::fmt::Debug,
{
    pub fn new(star: Star<T>, dnn: Vec<Affine<T>>) -> Self {
        let mut out = Constellation {
            arena: Vec::new(),
            dnn_affines: dnn,
        };
        let star_node = StarNode {
            idx: 0,
            parent: None,
            children: None,
            star: star,
            layer: 0,
            remaining_steps: None,
        };
        out.arena.push(star_node);
        out
    }

    pub fn sample(
        &mut self,
        loc: Array1<T>,
        scale: Array2<T>,
        cdf_samples: usize,
        num_samples: usize,
    ) -> Vec<Array1<f64>> {
        let mut current_node = 0;
        let mut rng = rand::thread_rng();
        loop {
            // Expand
            let children = self.expand_node(current_node);
            self.arena[current_node].children = Some(children.clone());
            if children.is_empty() {
                // sample from leaf
                println!("{:?}", self.arena[current_node].star);
                let samples =
                    self.arena[current_node]
                        .star
                        .trunc_gaussian_sample(&loc, &scale, num_samples);
                return samples;
            } else if children.len() == 1 {
                // continue with only child
                current_node = children[0];
            } else {
                // Bernoulli trial to choose child
                // In this branch, there can only be exactly two children
                let a_out =
                    self.arena[children[0]]
                        .star
                        .trunc_gaussian_cdf(&loc, &scale, cdf_samples);
                let b_out =
                    self.arena[children[1]]
                        .star
                        .trunc_gaussian_cdf(&loc, &scale, cdf_samples);
                let mut a = a_out.0;
                let mut b = b_out.0;
                println!("a {} b {}", a, b);
                current_node = if Bernoulli::new(a / (a + b)).unwrap().sample(&mut rng) {
                    println!("a");
                    children[0]
                } else {
                    println!("b");
                    children[1]
                };
            }
        }
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

    fn expand_node(&mut self, idx: usize) -> Vec<usize> {
        let node = &self.arena[idx];
        let node_children = &node.children;
        let node_idx = node.idx;
        let node_layer = node.layer;
        let node_remaining_steps = node.remaining_steps;
        if let Some(childs) = node_children {
            // if children exist already, return them
            childs.clone()
        } else {
            // check if there is a step relu to do
            if let Some(remaining_steps) = node_remaining_steps {
                let new_child_stars = node.star.step_relu(remaining_steps);
                let new_remaining_steps = if remaining_steps == 0 {
                    None
                } else {
                    Some(remaining_steps - 1)
                };
                new_child_stars
                    .into_iter()
                    .map(|x| self.add_node(x, node_idx, node_layer, new_remaining_steps))
                    .collect()
            } else {
                // check if there's another affine to do
                if node_layer < self.dnn_affines.len() {
                    let affine = &self.dnn_affines[node.layer];
                    let child_star = node.star.clone().affine_map(&affine);
                    let repr_space_dim = Some(child_star.representation_space_dim() - 1);
                    vec![self.add_node(child_star, node.idx, node.layer + 1, repr_space_dim)]
                } else {
                    // No step relus and no layers remaining means this is a leaf
                    Vec::new()
                }
            }
        }
    }
}

#[derive(Debug)]
struct StarNode<T: num::Float> {
    idx: usize,
    parent: Option<usize>,
    children: Option<Vec<usize>>,
    star: Star<T>,
    layer: usize,
    remaining_steps: Option<usize>,
}

impl<T: num::Float> StarNode<T> {
    fn new(
        idx: usize,
        parent: Option<usize>,
        star: Star<T>,
        layer: usize,
        remaining_steps: Option<usize>,
    ) -> Self {
        Self {
            idx: idx,
            parent: parent,
            star: star,
            children: None,
            layer: layer,
            remaining_steps: remaining_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    extern crate ndarray_rand;

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
                Array1::zeros(out_), //Array1::random(out_, dist),
            )
        };
        let star = Star::default(2);
        let a = generate_layer(2, 4);
        let b = generate_layer(4, 2);
        let c = generate_layer(2, 3);
        let mut constellation = Constellation::new(star, vec![a, b, c]);

        let loc = Array1::zeros(2);
        let scale = Array2::eye(2);
        let val = constellation.sample(loc, scale, 10000, 10);
        println!("{:?}", val);
        assert_eq!(0, 1);
    }
}
