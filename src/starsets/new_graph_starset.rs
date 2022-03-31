use super::new_starset::{StarId, StarRelationship, StarRelationshipId, StarSet, StarSet2};
use crate::bounds::Bounds1;
use crate::dnn::DNN;
use crate::graph::RepresentationId;
use crate::star::Star;
use ndarray::Dimension;
use ndarray::Ix2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStarset<D: 'static + Dimension> {
    arena: Vec<Star<D>>,
    star_representation_map: Vec<RepresentationId>,
    dnn: DNN,
}

impl<D: Dimension> GraphStarset<D> {
    fn new(dnn: DNN, input_star: Star<D>) -> Self {
        Self {
            arena: vec![input_star],
            star_representation_map: vec![RepresentationId::new(0, None)],
            dnn,
        }
    }
}

impl<D: 'static + Dimension> StarSet<D> for GraphStarset<D> {
    fn get_root_id(&self) -> StarId {
        0
    }

    fn get_star_representation_id(&self, star_id: usize) -> RepresentationId {
        self.star_representation_map[star_id]
    }

    fn get_graph(&self) -> &crate::graph::Graph {
        self.dnn.get_graph()
    }

    fn get_star(&self, star_id: StarId) -> &Star<D> {
        todo!()
    }

    fn get_relationship(&self, relationship_id: StarRelationshipId) -> &StarRelationship {
        todo!()
    }

    fn add_star(&mut self, star: Star<D>) -> StarId {
        todo!()
    }

    fn add_relationship(
        &mut self,
        star_rel: super::new_starset::StarRelationship,
    ) -> StarRelationshipId {
        todo!()
    }
}

impl StarSet2 for GraphStarset<Ix2> {
    fn get_input_dim(&self) -> usize {
        self.arena[0].input_dim()
    }

    /// Get the fixed part of the input
    fn get_input_bounds(&self, star_id: StarId) -> Bounds1 {
        todo!();
    }

    /// TODO: Implement with a cache because it is expensive
    fn get_axis_aligned_input_bounds(&self, star_id: StarId, outer_bounds: &Bounds1) -> &Bounds1 {
        todo!();
    }
}
