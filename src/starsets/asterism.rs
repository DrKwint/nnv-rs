use crate::bounds::Bounds1;
use crate::dnn::dnn::DNN;
use crate::dnn::dnn_iter::{DNNIndex, DNNIterator};
use crate::star::Star;
use crate::star_node::StarNode;
use crate::star_node::StarNodeType;
use crate::starsets::CensoredProbStarSet2;
use crate::starsets::ProbStarSet;
use crate::starsets::ProbStarSet2;
use crate::starsets::StarSet;
use crate::starsets::StarSet2;
use crate::util::ArenaLike;
use crate::NNVFloat;
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
    node_type: Vec<Option<StarNodeType>>,
    parents: Vec<Option<usize>>,
    feasible: Vec<Option<bool>>, // Nodes are assumed to be feasible until proven otherwise
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
            let initial_idx = DNNIterator::new(&dnn, DNNIndex::default()).next().unwrap();
            let star_node = StarNode::default(input_star, None, initial_idx);
            vec![star_node]
        };
        let node_type = vec![None];
        let parents = vec![None];
        let feasible = vec![None];
        Self {
            arena,
            node_type,
            parents,
            feasible,
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
    fn get_node_iter(&self) -> Self::NI<'_> {
        self.arena.iter()
    }

    type NTI<'a> = std::slice::Iter<'a, Option<StarNodeType>>;
    fn get_node_type_iter(&self) -> Self::NTI<'_> {
        self.node_type.iter()
    }

    fn add_node(&mut self, node: StarNode<D>, parent_id: usize) -> usize {
        let child_idx = self.arena.new_node(node);
        let fst_child_idx = self.node_type.new_node(None);
        let snd_child_idx = self.parents.new_node(Some(parent_id));
        let trd_child_idx = self.feasible.new_node(None);
        debug_assert_eq!(child_idx, fst_child_idx);
        debug_assert_eq!(child_idx, snd_child_idx);
        debug_assert_eq!(child_idx, trd_child_idx);
        child_idx
    }

    fn get_input_bounds(&self) -> &Option<crate::bounds::Bounds1> {
        &self.input_bounds_opt
    }

    fn get_dnn(&self) -> &DNN {
        &self.dnn
    }

    fn try_get_node_parent_id(&self, node_id: usize) -> Option<usize> {
        self.parents[node_id]
    }

    fn get_node_dnn_index(&self, node_id: usize) -> DNNIndex {
        self.arena[node_id].get_dnn_index()
    }

    fn try_get_node_type(&self, node_id: usize) -> &Option<StarNodeType> {
        &self.node_type[node_id]
    }

    fn set_node_type(&mut self, node_id: usize, children: StarNodeType) {
        self.node_type[node_id] = Some(children);
    }

    fn reset_with_star(&mut self, input_star: Star<D>, input_bounds_opt: Option<Bounds1>) {
        self.arena = {
            let initial_idx = DNNIterator::new(&self.dnn, DNNIndex::default())
                .next()
                .unwrap();
            let star_node = StarNode::default(input_star, None, initial_idx);
            vec![star_node]
        };
        self.node_type = vec![None];
        self.parents = vec![None];
        self.feasible = vec![None];
        self.input_bounds_opt = input_bounds_opt;
    }
}

impl StarSet2 for Asterism<Ix2> {
    /// Returns the children of a node
    ///
    /// Lazily loads children into the arena and returns a reference to them.
    ///
    /// # Arguments
    ///
    /// * `self` - The node to expand
    /// * `node_arena` - The data structure storing star nodes
    /// * `dnn_iter` - The iterator of operations in the dnn
    ///
    /// # Returns
    /// * `children` - `StarNodeType<T>`
    ///
    /// # Panics
    fn get_node_type(&mut self, node_id: usize) -> &StarNodeType {
        if self
            .node_type
            .get(node_id)
            .and_then(std::option::Option::as_ref)
            .is_some()
        {
            self.node_type
                .get(node_id)
                .and_then(std::option::Option::as_ref)
                .unwrap()
        } else {
            self.expand(node_id)
        }
    }
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

    fn set_scale(&mut self, val: Array2<NNVFloat>) {
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
        #[test]
        fn test_sample_safe_star(mut asterism in generic_asterism(2, 2, 2, 2)) {
            let mut rng = rand::thread_rng();
            let _default: Array1<f64> = Array1::zeros(asterism.get_dnn().input_shape()[0].unwrap());
            let _sample = asterism.sample_safe_star(1, &mut rng, None);
        }


        #[test]
        fn test_dfs_samples(mut asterism in generic_asterism(2, 2, 2, 2)) {
            let num_samples = 4;
            let time_limit_opt = None;

            let mut rng = rand::thread_rng();
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
