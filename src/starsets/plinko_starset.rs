use super::starset::StarId;
use super::starset::StarSet2;
use crate::graph::GraphState;
use crate::NNVFloat;
use ndarray::Array1;
use std::collections::hash_map::Iter;
use std::collections::HashMap;
use std::hash::Hash;

type SuperStarId = usize;

struct SuperStarSet<SS: StarSet2> {
    starset: SS,
    frontier: Vec<SuperStar>,
}

impl<SS: StarSet2> SuperStarSet<SS> {
    fn new(starset: SS) -> Self {
        assert_eq!(starset.get_dnn().get_output_representation_ids().len(), 1);
        assert_eq!(starset.get_dnn().get_input_representation_ids().len(), 1);
        let frontier = vec![SuperStar::new(
            vec![starset.get_dnn().get_output_representation_ids()[0]],
            vec![(
                starset.get_dnn().get_input_representation_ids()[0],
                starset.get_root_id(),
            )]
            .into_iter()
            .collect(),
            starset.get_dnn().get_graph(),
        )];
        Self { starset, frontier }
    }

    fn get_root_id(&self) -> SuperStarId {
        todo!()
    }

    fn expand(&self, superstar_id: SuperStarId) -> Vec<SuperStarId> {
        todo!()
    }
}

struct MultiRepData {
    output_value: NNVFloat,
    activation_pattern: Vec<usize>,
}

type SuperStar = GraphState<StarId>;

impl SuperStar {
    // EEN
    fn expand(&self, starset: &impl StarSet2) -> Vec<Self> {
        // Run the operation to get a set of output stars. Add each star to the starset.

        // Apply the operation to the graphstate to get the next graphstate.
        todo!()
    }
}

struct StratifiedSamplingFrontier<Id, T> {
    nodes_samples: HashMap<Id, Vec<T>>,
}

impl<Id, T> StratifiedSamplingFrontier<Id, T>
where
    Id: Hash + Eq,
{
    fn new(root: Id, root_samples: Vec<T>) -> Self {
        StratifiedSamplingFrontier {
            nodes_samples: {
                let hm = HashMap::new();
                hm.insert(root, root_samples);
                hm
            },
        }
    }

    fn replace(&mut self, parent: Id, children: Vec<(Id, Vec<T>)>) {
        todo!()
    }

    fn star_sample_mean(&self, star_id: Id) -> NNVFloat {
        todo!()
    }

    fn star_sample_variance(&self, star_id: Id) -> NNVFloat {
        todo!()
    }

    fn iter(&self) -> Iter<'_, Id, Vec<T>> {
        self.nodes_samples.iter()
    }
}

// Structure that tracks, at each split in the tree, how many samples went to each node from their parent
// This allows us to recover the probability of each region in the partition by taking the product of the probabilities from root to node.
struct PlinkoProbabilities {
    drop_count: HashMap<StarId, Vec<(StarId, usize)>>, // Maps parent to children and how many samples went to each child&&
}

impl PlinkoProbabilities {
    fn drop(&mut self, src_node: StarId, dst_node: StarId) {}

    fn star_probability(&self, star_id: StarId) -> NNVFloat {
        todo!()
    }
}

fn the_whole_enchilada<SS: StarSet2>(
    superstarset: SuperStarSet<SS>,
    frontier_select: impl Fn(&StratifiedSamplingFrontier<SuperStarId, MultiRepData>) -> SuperStarId,
    sample_node: impl Fn(SuperStarId, usize) -> Vec<MultiRepData>,
    budget: usize,
) {
    // Iterative stratified importance sampling
    // 0. Initialize the running estimated data
    let mut fuel = budget;
    // 1. Initialize frontier
    let mut frontier = StratifiedSamplingFrontier::new(
        superstarset.get_root_id(),
        sample_node(superstarset.get_root_id(), 100),
    );

    loop {
        // 2. Run selection function to take a node from the frontier to expand and sample
        // 2a. Expand
        let parent_id = frontier_select(&frontier);
        let children_nodes: Vec<StarId> = superstarset.expand(parent_id);
        // 2b. Sample from each node
        // 2ai. Calculate sample budget for each node
        let children_samples: Vec<(SuperStarId, Vec<MultiRepData>)> = children_nodes
            .into_iter()
            .map(|child_id: StarId| (child_id, sample_node(child_id, 100)))
            .collect();
        // 2aii. Add new samples to the frontier
        frontier.replace(parent_id, children_samples);

        // 3. Update the running estimate of the values

        fuel -= 1;
        if fuel <= 0 {
            break;
        }
    }
    // Wrapup code
}

fn select_max_variance<Id: Hash + Eq, T>(frontier: &StratifiedSamplingFrontier<Id, T>) -> Id {
    frontier.iter();
    todo!()
}
