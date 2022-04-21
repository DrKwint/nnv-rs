use super::starset::{StarId, StarRelationship, StarRelationshipId, StarSet, StarSet2};
use crate::bounds::{Bounds, Bounds1};
use crate::dnn::DNN;
use crate::graph::{Graph, RepresentationId};
use crate::star::Star;
use itertools::Itertools;
use ndarray::Dimension;
use ndarray::Ix2;
use std::cell::{Ref, RefCell};

pub struct GraphStarset<D: 'static + Dimension> {
    /// The network for which the starset is generated
    dnn: DNN,
    /// The input bounds of the star set
    input_bounds: Bounds<D>,

    // Parallel arrays
    /// Storage structure for stars
    arena: RefCell<Vec<Star<D>>>,
    /// The RepresentationId that each star represents
    representations: RefCell<Vec<RepresentationId>>,
    /// Output bounds of a prefix network up to and including the operation the produces the star
    local_output_bounds: RefCell<Vec<Option<Bounds<D>>>>,

    /// The relationships between stars, includes the associated graph operation
    relationships: RefCell<Vec<StarRelationship>>,
}

impl<D: Dimension> GraphStarset<D> {
    pub fn new(dnn: DNN, input_bounds: Bounds<D>, input_star: Star<D>) -> Self {
        let local_output_bounds = if input_star.num_constraints() == 0 {
            input_bounds.clone()
        } else {
            todo!();
        };

        Self {
            dnn,
            input_bounds,
            arena: RefCell::new(vec![input_star]),
            representations: RefCell::new(vec![RepresentationId::new(0, None)]),
            local_output_bounds: RefCell::new(vec![Some(local_output_bounds)]),
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

    fn get_input_bounds(&self) -> &Bounds<D> {
        &self.input_bounds
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

    fn get_producing_relationship(&self, star_id: &StarId) -> Option<StarRelationship> {
        self.relationships
            .borrow()
            .iter()
            .find(|rel| rel.output_star_ids.iter().flatten().contains(star_id))
            .map(|x| x.clone())
    }

    fn get_relationship(&self, relationship_id: StarRelationshipId) -> Ref<StarRelationship> {
        assert!(relationship_id < self.relationships.borrow().len());
        Ref::map(self.relationships.borrow(), |vec| &vec[relationship_id])
    }

    fn add_star(
        &self,
        star: Star<D>,
        representation_id: RepresentationId,
        local_output_bounds: Option<Bounds<D>>,
    ) -> StarId {
        let star_id = self.arena.borrow().len();
        self.arena.borrow_mut().push(star);
        self.representations.borrow_mut().push(representation_id);
        self.local_output_bounds
            .borrow_mut()
            .push(local_output_bounds);
        assert_eq!(
            self.arena.borrow().len(),
            self.representations.borrow().len()
        );
        assert_eq!(
            self.arena.borrow().len(),
            self.local_output_bounds.borrow().len()
        );
        star_id
    }

    fn add_relationship(&self, star_rel: super::starset::StarRelationship) -> StarRelationshipId {
        // TODO: Do checks about relationship to cache properties
        let rel_id = self.relationships.borrow().len();
        self.relationships.borrow_mut().push(star_rel);
        rel_id
    }

    /// Get all stars that represent the transformation to reach a specific representation
    fn get_stars_for_representation(&self, repr_id: &RepresentationId) -> Vec<StarId> {
        self.representations
            .borrow()
            .iter()
            .enumerate()
            .filter(|(_, &star_repr_id)| star_repr_id == *repr_id)
            .map(|(star_id, _)| star_id)
            .collect()
    }
}

impl StarSet2 for GraphStarset<Ix2> {
    fn get_input_dim(&self) -> usize {
        self.arena.borrow()[0].input_dim()
    }

    /// TODO: Implement with a cache because it is expensive
    fn get_local_output_bounds(&self, star_id: StarId) -> Ref<Option<Bounds1>> {
        assert!(star_id < self.local_output_bounds.borrow().len());
        Ref::map(self.local_output_bounds.borrow(), |vec| &vec[star_id])
    }
}
