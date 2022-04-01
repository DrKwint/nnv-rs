use super::new_starset::{StarId, StarRelationship, StarRelationshipId, StarSet, StarSet2};
use crate::bounds::{Bounds, Bounds1};
use crate::dnn::DNN;
use crate::graph::RepresentationId;
use crate::star::Star;
use ndarray::Dimension;
use ndarray::Ix2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStarset<D: 'static + Dimension> {
    /// The network for which the starset is generated
    dnn: DNN,
    // Parallel arrays
    /// Storage structure for stars
    arena: Vec<Star<D>>,
    /// The RepresentationId that each star represents
    representations: Vec<RepresentationId>,
    /// Axis aligned input bounds for each star
    input_bounds: Vec<Bounds<D>>,

    /// The relationships between stars, includes the associated graph operation
    relationships: Vec<StarRelationship>,
}

impl<D: Dimension> GraphStarset<D> {
    pub fn new(dnn: DNN, input_star: Star<D>, input_bounds: Bounds<D>) -> Self {
        Self {
            dnn,
            arena: vec![input_star],
            representations: vec![RepresentationId::new(0, None)],
            input_bounds: vec![input_bounds],
            relationships: vec![],
        }
    }
}

impl<D: 'static + Dimension> StarSet<D> for GraphStarset<D> {
    fn get_root_id(&self) -> StarId {
        0
    }

    fn get_star_representation_id(&self, star_id: usize) -> RepresentationId {
        self.representations[star_id]
    }

    fn get_graph(&self) -> &crate::graph::Graph {
        self.dnn.get_graph()
    }

    fn get_star(&self, star_id: StarId) -> &Star<D> {
        assert!(star_id < self.arena.len());
        &self.arena[star_id]
    }

    fn get_relationship(&self, relationship_id: StarRelationshipId) -> &StarRelationship {
        assert!(relationship_id < self.relationships.len());
        &self.relationships[relationship_id]
    }

    fn add_star(
        &mut self,
        star: Star<D>,
        representation_id: RepresentationId,
        axis_aligned_input_bounds: Bounds<D>,
    ) -> StarId {
        let star_id = self.arena.len();
        self.arena.push(star);
        self.representations.push(representation_id);
        self.input_bounds.push(axis_aligned_input_bounds);
        assert_eq!(self.arena.len(), self.representations.len());
        star_id
    }

    fn add_relationship(
        &mut self,
        star_rel: super::new_starset::StarRelationship,
    ) -> StarRelationshipId {
        // TODO: Do checks about relationship to cache properties
        let rel_id = self.relationships.len();
        self.relationships.push(star_rel);
        rel_id
    }
}

impl StarSet2 for GraphStarset<Ix2> {
    fn get_input_dim(&self) -> usize {
        self.arena[0].input_dim()
    }

    /// TODO: Implement with a cache because it is expensive
    fn get_axis_aligned_input_bounds(&self, star_id: StarId) -> &Bounds1 {
        assert!(star_id < self.input_bounds.len());
        &self.input_bounds[star_id]
    }
}
