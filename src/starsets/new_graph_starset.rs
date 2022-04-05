use std::cell::{Ref, RefCell};

use super::new_starset::{StarId, StarRelationship, StarRelationshipId, StarSet, StarSet2};
use crate::bounds::{Bounds, Bounds1};
use crate::dnn::DNN;
use crate::graph::{Graph, RepresentationId};
use crate::star::Star;
use ndarray::Dimension;
use ndarray::Ix2;

pub struct GraphStarset<D: 'static + Dimension> {
    /// The network for which the starset is generated
    dnn: DNN,
    // Parallel arrays
    /// Storage structure for stars
    arena: RefCell<Vec<Star<D>>>,
    /// The RepresentationId that each star represents
    representations: RefCell<Vec<RepresentationId>>,
    /// Axis aligned input bounds for each star
    input_bounds: RefCell<Vec<Bounds<D>>>,

    /// The relationships between stars, includes the associated graph operation
    relationships: RefCell<Vec<StarRelationship>>,
}

impl<D: Dimension> GraphStarset<D> {
    pub fn new(dnn: DNN, input_star: Star<D>, input_bounds: Bounds<D>) -> Self {
        Self {
            dnn,
            arena: RefCell::new(vec![input_star]),
            representations: RefCell::new(vec![RepresentationId::new(0, None)]),
            input_bounds: RefCell::new(vec![input_bounds]),
            relationships: RefCell::new(vec![]),
        }
    }
}

impl<D: 'static + Dimension> StarSet<D> for GraphStarset<D> {
    fn get_graph(&self) -> &Graph {
        self.dnn.get_graph()
    }

    fn get_dnn(&self) -> &DNN {
        &self.dnn
    }

    fn get_root_id(&self) -> StarId {
        0
    }

    fn get_star_representation_id(&self, star_id: usize) -> RepresentationId {
        self.representations.borrow()[star_id]
    }

    fn get_star(&self, star_id: StarId) -> Ref<Star<D>> {
        assert!(star_id < self.arena.borrow().len());
        Ref::map(self.arena.borrow(), |vec| &vec[star_id])
    }

    fn get_relationship(&self, relationship_id: StarRelationshipId) -> Ref<StarRelationship> {
        assert!(relationship_id < self.relationships.borrow().len());
        Ref::map(self.relationships.borrow(), |vec| &vec[relationship_id])
    }

    fn add_star(
        &self,
        star: Star<D>,
        representation_id: RepresentationId,
        axis_aligned_input_bounds: Bounds<D>,
    ) -> StarId {
        let star_id = self.arena.borrow().len();
        self.arena.borrow_mut().push(star);
        self.representations.borrow_mut().push(representation_id);
        self.input_bounds
            .borrow_mut()
            .push(axis_aligned_input_bounds);
        assert_eq!(
            self.arena.borrow().len(),
            self.representations.borrow().len()
        );
        star_id
    }

    fn add_relationship(
        &self,
        star_rel: super::new_starset::StarRelationship,
    ) -> StarRelationshipId {
        // TODO: Do checks about relationship to cache properties
        let rel_id = self.relationships.borrow().len();
        self.relationships.borrow_mut().push(star_rel);
        rel_id
    }
}

impl StarSet2 for GraphStarset<Ix2> {
    fn get_input_dim(&self) -> usize {
        self.arena.borrow()[0].input_dim()
    }

    /// TODO: Implement with a cache because it is expensive
    fn get_axis_aligned_input_bounds(&self, star_id: StarId) -> Ref<Bounds1> {
        assert!(star_id < self.input_bounds.borrow().len());
        Ref::map(self.input_bounds.borrow(), |vec| &vec[star_id])
    }
}
