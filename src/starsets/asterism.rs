use crate::bounds::Bounds1;
use crate::dnn::dnn::DNN;
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeRelationship;
use crate::star_node::StarNodeType;
use crate::starsets::CensoredProbStarSet2;
use crate::starsets::ProbStarSet;
use crate::starsets::ProbStarSet2;
use crate::starsets::StarSet;
use crate::starsets::StarSet2;
use crate::util::ArenaLike;
use crate::NNVFloat;
use itertools::Itertools;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::{Array1, Dimension, Ix2};
use serde::{Deserialize, Serialize};

use super::CensoredProbStarSet;

#[derive(Debug, Serialize, Deserialize)]
pub struct Asterism<D: Dimension> {
    // These are parallel arrays with an entry per node
    arena: Vec<StarNode<D>>,
    parents: Vec<Vec<usize>>,
    children: Vec<Option<Vec<usize>>>,
    producing_relationship: Vec<Option<usize>>,
    feasible: Vec<Option<bool>>, // Nodes are assumed to be feasible until proven otherwise/*  */
    // Relationships stores all relationships between nodes
    relationships: Vec<StarNodeRelationship>,
    // These are values global to the struct
    loc: Array1<NNVFloat>,
    scale: Array2<NNVFloat>,
    safe_value: NNVFloat,
    input_bounds_opt: Option<Bounds1>,
    dnn: DNN,
    max_accept_reject_iters: usize,
    num_cdf_samples: usize,
    stability_eps: NNVFloat,
}

impl<D: Dimension> Asterism<D> {
    pub fn new(
        dnn: DNN,
        input_star: Star<D>,
        loc: Array1<NNVFloat>,
        scale: Array2<NNVFloat>,
        safe_value: NNVFloat,
        input_bounds_opt: Option<Bounds1>,
        max_accept_reject_iters: usize,
        num_cdf_samples: usize,
        stability_eps: NNVFloat,
    ) -> Self {
        let arena = {
            let star_node = StarNode::default(input_star, None);
            vec![star_node]
        };
        let parents = vec![vec![]];
        let children = vec![None];
        let producing_relationship = vec![None];
        let feasible = vec![None];
        let relationships = vec![];
        Self {
            arena,
            parents,
            children,
            producing_relationship,
            feasible,
            relationships,
            loc,
            scale,
            safe_value,
            input_bounds_opt,
            dnn,
            max_accept_reject_iters,
            num_cdf_samples,
            stability_eps,
        }
    }
}

impl<D: 'static + Dimension> StarSet<D> for Asterism<D> {
    fn get_node(&self, node_id: usize) -> &StarNode<D> {
        &self.arena[node_id]
    }

    fn get_node_mut(&mut self, node_id: usize) -> &mut StarNode<D> {
        &mut self.arena[node_id]
    }

    type NI<'a> = std::slice::Iter<'a, StarNode<D>>;
    // fn get_node_iter(&self) -> Self::NI<'_> {
    //     self.arena.iter()
    // }

    type NTI<'a> = std::slice::Iter<'a, Option<StarNodeType>>;
    // fn get_node_type_iter(&self) -> Self::NTI<'_> {
    //     self.node_type.iter()
    // }

    fn add_node(&mut self, node: StarNode<D>) -> usize {
        let child_idx = self.arena.push_node(node);
        let fst_child_idx = self.parents.push_node(vec![]);
        let snd_child_idx = self.children.push_node(None);
        let trd_child_idx = self.producing_relationship.push_node(None);
        let fth_child_idx = self.feasible.push_node(None);
        debug_assert_eq!(child_idx, fst_child_idx);
        debug_assert_eq!(child_idx, snd_child_idx);
        debug_assert_eq!(child_idx, trd_child_idx);
        child_idx
    }

    fn add_node_relationship(&mut self, rel: StarNodeRelationship) -> usize {
        let rel_id = self.relationships.push_node(rel);
        rel.output_node_ids
            .unwrap()
            .iter()
            .for_each(|&output_id| self.producing_relationship[output_id] = Some(rel_id));

        rel.input_node_ids
            .iter()
            .cartesian_product(rel.output_node_ids.unwrap().iter())
            .for_each(|(&input_id, &output_id)| {
                self.parents[output_id].push(input_id);
                if self.children.get(input_id).is_none() {
                    self.children[input_id] = Some(vec![]);
                }
                self.children[input_id].unwrap().push(output_id);
            });
        rel_id
    }

    fn get_input_bounds(&self) -> &Option<crate::bounds::Bounds1> {
        &self.input_bounds_opt
    }

    fn get_dnn(&self) -> &DNN {
        &self.dnn
    }

    fn try_get_node_parent_ids(&self, node_id: usize) -> Option<&Vec<usize>> {
        self.parents.get(node_id)
    }

    fn reset_with_star(&mut self, input_star: Star<D>, input_bounds_opt: Option<Bounds1>) {
        self.arena = {
            let star_node = StarNode::default(input_star, None);
            vec![star_node]
        };
        self.parents = vec![vec![]];
        self.feasible = vec![None];
        self.input_bounds_opt = input_bounds_opt;
    }

    fn get_parent_nodes(&self, node_id: usize) -> Vec<&StarNode<D>> {
        self.parents
            .get(node_id)
            .unwrap_or(&vec![])
            .into_iter()
            .map(|&parent_id| self.get_node(parent_id))
            .collect::<Vec<_>>()
    }

    fn get_child_nodes(&self, node_id: usize) -> Option<Vec<&StarNode<D>>> {
        self.children
            .get(node_id)
            .map(|child_ids_opt| {
                child_ids_opt.map(|child_ids| {
                    child_ids
                        .into_iter()
                        .map(|child_id| self.get_node(child_id))
                        .collect::<Vec<_>>()
                })
            })
            .unwrap_or(None)
    }

    fn get_parent_ids(&self, node_id: usize) -> Vec<usize> {
        self.parents[node_id]
    }

    fn get_child_ids(&self, node_id: usize) -> Option<Vec<usize>> {
        self.children[node_id]
    }

    fn get_creating_relationship(
        &self,
        node_id: usize,
    ) -> Option<&crate::star_node::StarNodeRelationship> {
        todo!()
    }

    fn get_node_relationship(&self, rel_id: usize) -> &crate::star_node::StarNodeRelationship {
        todo!()
    }

    fn get_node_relationship_mut(
        &self,
        rel_id: usize,
    ) -> &mut crate::star_node::StarNodeRelationship {
        todo!()
    }

    fn get_node_producing_relationship_id(&self, node_id: usize) -> usize {
        todo!()
    }

    fn get_operation(&self, op_id: &crate::graph::OperationId) -> &dyn crate::graph::Operation {
        todo!()
    }

    fn get_operation_node(
        &self,
        op_id: &crate::graph::OperationId,
    ) -> &crate::graph::OperationNode {
        todo!()
    }

    fn get_root_id(&self) -> usize {
        0
    }
}

impl StarSet2 for Asterism<Ix2> {
    // Returns the children of a node
    //
    // Lazily loads children into the arena and returns a reference to them.
    //
    // # Arguments
    //
    // * `self` - The node to expand
    // * `node_arena` - The data structure storing star nodes
    // * `dnn_iter` - The iterator of operations in the dnn
    //
    // # Returns
    // * `children` - `StarNodeType<T>`
    //
    // # Panics
    // fn get_node_type(&mut self, node_id: usize) -> &StarNodeType {
    //     if self
    //         .node_type
    //         .get(node_id)
    //         .and_then(std::option::Option::as_ref)
    //         .is_some()
    //     {
    //         self.node_type
    //             .get(node_id)
    //             .and_then(std::option::Option::as_ref)
    //             .unwrap()
    //     } else {
    //         self.expand(node_id)
    //     }
    // }

    // fn get_node_type_mut(&mut self, node_id: usize) -> &mut StarNodeType {
    //     if self
    //         .node_type
    //         .get(node_id)
    //         .and_then(std::option::Option::as_ref)
    //         .is_some()
    //     {
    //         self.node_type
    //             .get_mut(node_id)
    //             .and_then(std::option::Option::as_mut)
    //             .unwrap()
    //     } else {
    //         self.expand(node_id)
    //     }
    // }
}

impl<D: 'static + Dimension> ProbStarSet<D> for Asterism<D> {
    fn reset_input_distribution(&mut self, loc: Array1<NNVFloat>, scale: Array2<NNVFloat>) {
        self.loc = loc;
        self.scale = scale;
        self.arena.iter_mut().for_each(StarNode::reset_cdf);
    }
}

impl ProbStarSet2 for Asterism<Ix2> {
    fn get_node_mut_with_borrows(
        &mut self,
        node_id: usize,
    ) -> (
        &mut StarNode<Ix2>,
        ArrayView1<NNVFloat>,
        ArrayView2<NNVFloat>,
        &DNN,
    ) {
        (
            &mut self.arena[node_id],
            self.loc.view(),
            self.scale.view(),
            &self.dnn,
        )
    }

    fn get_loc(&self) -> ArrayView1<NNVFloat> {
        self.loc.view()
    }

    fn set_loc(&mut self, val: Array1<NNVFloat>) {
        self.loc = val;
    }

    fn get_scale(&self) -> ArrayView2<NNVFloat> {
        self.scale.view()
    }

    fn set_scale(&mut self, val: Vec<Array2<NNVFloat>>) {
        self.scale = val;
    }

    fn get_max_accept_reject_iters(&self) -> usize {
        self.max_accept_reject_iters
    }

    fn get_stability_eps(&self) -> NNVFloat {
        self.stability_eps
    }

    fn get_cdf_samples(&self) -> usize {
        self.num_cdf_samples
    }
}

impl<D: 'static + Dimension> CensoredProbStarSet<D> for Asterism<D> {
    type I<'a> = std::slice::Iter<'a, std::option::Option<bool>>;

    fn get_safe_value(&self) -> NNVFloat {
        self.safe_value
    }

    fn set_safe_value(&mut self, val: NNVFloat) {
        self.safe_value = val;
    }

    fn get_feasible_iter(&self) -> Self::I<'_> {
        self.feasible.iter()
    }

    fn is_node_infeasible(&self, id: usize) -> bool {
        matches!(self.feasible[id], Some(false))
    }

    fn get_node_feasibility(&self, id: usize) -> Option<bool> {
        self.feasible[id]
    }

    fn set_node_feasibility(&mut self, id: usize, val: bool) {
        self.feasible[id] = Some(val);
    }
}

impl CensoredProbStarSet2 for Asterism<Ix2> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::starsets::CensoredProbStarSet2;
    use crate::test_util::*;
    // use log::LevelFilter;
    // use log4rs::append::file::FileAppender;
    // use log4rs::config::{Appender, Config, Root};
    // use log4rs::encode::pattern::PatternEncoder;
    use proptest::*;
    use rand::prelude::*;
    use std::fs;
    // use std::sync::Once;

    // static INIT: Once = Once::new();

    // fn setup() {
    //     INIT.call_once(|| {
    //         let logfile = FileAppender::builder()
    //             .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
    //             .build("log/output.log")
    //             .unwrap();

    //         let config = Config::builder()
    //             .appender(Appender::builder().build("logfile", Box::new(logfile)))
    //             .build(Root::builder().appender("logfile").build(LevelFilter::Info))
    //             .unwrap();

    //         let _log_res = log4rs::init_config(config);
    //     });
    // }

    proptest! {
        /// Test that a sample and a fast forward bounds check from the root have the same feasbility
        #[test]
        fn test_feasibility(mut asterism in generic_asterism(2,2,2,2)) {
            let eps = 1e-5;
            let mut rng = SmallRng::seed_from_u64(0);
            let (lower_bound, upper_bound) = asterism.get_node_output_bounds(asterism.get_root_id());
            let root_samples = asterism.sample_root_node(100, &mut rng);
            let sample_output_vals = root_samples.columns().into_iter()
                    .map(|x| asterism.get_dnn().forward1(x.to_owned()));
            prop_assert!(sample_output_vals.clone().all(|val| *val.shape() == [1]));
            let input_bounds = asterism.get_input_bounds().as_ref().cloned().unwrap();
            for (val, sample) in sample_output_vals.zip(root_samples.columns().into_iter()) {
                if !input_bounds.is_member(&sample) {
                    continue;
                }
                if input_bounds.lower().abs_diff_eq(&sample, eps) || input_bounds.upper().abs_diff_eq(&sample, eps) {
                    continue;
                }
                asterism.get_dnn().get_layers().iter().fold(sample.to_owned(), |x, layer| {
                    let repr = layer.forward1(&x);
                    repr
                });
                prop_assert!(val[[0]] >= lower_bound - eps && val[[0]] <= upper_bound + eps, "val: {}", val[[0]]);
            }
        }

        #[test]
        fn test_sample_safe_star(mut asterism in generic_asterism(2, 2, 2, 2)) {
            let mut rng = SmallRng::seed_from_u64(0);
            let _default: Array1<f64> = Array1::zeros(asterism.get_dnn().input_shape()[0].unwrap());
            let _sample = asterism.sample_safe_star(1, &mut rng, None);
        }


        #[test]
        fn test_dfs_samples(mut asterism in generic_asterism(2, 2, 2, 2)) {
            let num_samples = 4;
            let time_limit_opt = None;

            let mut rng = SmallRng::seed_from_u64(0);
            asterism.dfs_samples(num_samples, &mut rng, time_limit_opt);
        }

        #[test]
        fn test_serialization(mut asterism in generic_asterism(2,2,2,2)) {
            let num_samples = 4;
            let time_limit_opt = None;

            let mut rng = rand::thread_rng();
            asterism.dfs_samples(num_samples, &mut rng, time_limit_opt);

            let serialization = serde_json::to_string(&asterism).unwrap();
            let _serial_asterism: Asterism<Ix2> = serde_json::from_str(&serialization)?;
            fs::write("test.json", &serialization).expect("Unable to write file.");
        }
    }
}
