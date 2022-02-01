use crate::dnn::DNN;
use crate::gaussian::GaussianDistribution;
use crate::star_node::StarNode;
use crate::starsets::StarSet;
use crate::starsets::StarSet2;
use crate::NNVFloat;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::Rng;

pub trait ProbStarSet<D: 'static + Dimension>: StarSet<D> {
    fn reset_input_distribution(&mut self, loc: Array1<NNVFloat>, scale: Array2<NNVFloat>);
}

pub trait ProbStarSet2: ProbStarSet<Ix2> + StarSet2 {
    fn get_node_mut_with_borrows(
        &mut self,
        node_id: usize,
    ) -> (
        &mut StarNode<Ix2>,
        ArrayView1<NNVFloat>,
        ArrayView2<NNVFloat>,
        &DNN,
    );
    fn get_loc(&self) -> ArrayView1<NNVFloat>;
    fn set_loc(&mut self, val: Array1<NNVFloat>);
    fn get_scale(&self) -> ArrayView2<NNVFloat>;
    fn set_scale(&mut self, val: Array2<NNVFloat>);
    fn get_max_accept_reject_iters(&self) -> usize;
    fn get_stability_eps(&self) -> NNVFloat;
    fn get_cdf_samples(&self) -> usize;
    fn get_node_cdf<R: Rng>(&mut self, node_id: usize, rng: &mut R) -> NNVFloat {
        let cdf_samples = self.get_cdf_samples();
        let iters = self.get_max_accept_reject_iters();
        let stab_eps = self.get_stability_eps();
        let input_bounds_opt = self.get_input_bounds().clone();
        let (node_mut, loc, scale, dnn) = self.get_node_mut_with_borrows(node_id);
        node_mut.gaussian_cdf(
            loc,
            scale,
            cdf_samples,
            iters,
            rng,
            stab_eps,
            &input_bounds_opt,
        )
    }

    fn add_node_cdf(&mut self, node_id: usize, cdf: NNVFloat) {
        self.get_node_mut(node_id).add_cdf(cdf);
    }

    fn set_node_cdf(&mut self, node_id: usize, cdf: NNVFloat) {
        self.get_node_mut(node_id).set_cdf(cdf);
    }

    fn try_get_node_gaussian_distribution(&self, node_id: usize) -> Option<&GaussianDistribution> {
        self.get_node(node_id).try_get_gaussian_distribution()
    }

    fn sample_root_node<R: Rng>(&self, num_samples: usize, rng: &mut R) -> Array2<NNVFloat> {
        let input_dim = self.get_loc().len();
        let mut sample = Array2::random_using((num_samples, input_dim), StandardNormal, rng)
            * self.get_scale().diag()
            + self.get_loc();
        sample.swap_axes(0, 1);
        sample
    }

    fn get_node_gaussian_distribution(&mut self, node_id: usize) -> &mut GaussianDistribution {
        let max_accept_reject_iters = self.get_max_accept_reject_iters();
        let stability_eps = self.get_stability_eps();
        let input_bounds_opt = self.get_input_bounds().clone();
        let (node_mut, loc, scale, dnn) = self.get_node_mut_with_borrows(node_id);
        node_mut.get_gaussian_distribution(
            loc,
            scale,
            max_accept_reject_iters,
            stability_eps,
            &input_bounds_opt,
        )
    }

    /// # Panics
    fn initialize_node_tilting_from_parent(
        &mut self,
        node_id: usize,
        max_accept_reject_iters: usize,
        stability_eps: NNVFloat,
    ) {
        let parent_id = self.try_get_node_parent_id(node_id).unwrap();
        let parent_tilting_soln = {
            debug_assert!(
                self.try_get_node_gaussian_distribution(parent_id).is_some(),
                "parent_id: {}",
                parent_id
            );
            let parent_distr = self.try_get_node_gaussian_distribution(parent_id).unwrap();
            // This isn't always the case because the parent may be unconstrained (if it's the root)
            //debug_assert!(parent_distr.try_get_tilting_solution().is_some());
            parent_distr.try_get_tilting_solution().cloned()
        };
        let loc = self.get_loc().to_owned();
        let scale = self.get_scale().to_owned();
        let input_bounds_opt = self.get_input_bounds().clone();
        let child_distr = self.get_node_mut(node_id).get_gaussian_distribution(
            loc.view(),
            scale.view(),
            max_accept_reject_iters,
            stability_eps,
            &input_bounds_opt,
        );
        child_distr.populate_tilting_solution(parent_tilting_soln.as_ref());
    }

    fn add_node_with_tilting_initialization(
        &mut self,
        node: StarNode<Ix2>,
        parent_id: usize,
    ) -> usize {
        let node_id = self.add_node(node, parent_id);
        self.initialize_node_tilting_from_parent(
            node_id,
            self.get_max_accept_reject_iters(),
            self.get_stability_eps(),
        );
        node_id
    }

    /// # Panics
    fn sample_gaussian_node<R: Rng>(
        &mut self,
        node_id: usize,
        rng: &mut R,
        n: usize,
    ) -> Vec<Array1<NNVFloat>> {
        let initialization_opt = self.try_get_node_parent_id(node_id).and_then(|parent_id| {
            self.get_node(parent_id)
                .try_get_gaussian_distribution()
                .unwrap()
                .try_get_tilting_solution()
                .cloned()
        });
        let loc = self.get_loc().to_owned();
        let scale = self.get_scale().to_owned();
        let stability_eps = self.get_stability_eps();
        let max_accept_reject_iters = self.get_max_accept_reject_iters();
        let input_bounds_opt = self.get_input_bounds().clone();
        self.get_node_mut(node_id).gaussian_sample(
            rng,
            loc.view(),
            scale.view(),
            n,
            max_accept_reject_iters,
            initialization_opt.as_ref(),
            stability_eps,
            &input_bounds_opt,
        )
    }

    /// # Panics
    fn sample_gaussian_node_safe<R: Rng>(
        &mut self,
        node_id: usize,
        rng: &mut R,
        n: usize,
        max_iters: usize,
        safe_value: NNVFloat,
        stability_eps: NNVFloat,
    ) -> Vec<Array1<NNVFloat>> {
        let mut safe_star = self.get_node(node_id).get_safe_star(safe_value);
        let initialization_opt = self.try_get_node_parent_id(node_id).and_then(|parent_id| {
            self.get_node(parent_id)
                .try_get_gaussian_distribution()
                .unwrap()
                .try_get_tilting_solution()
        });
        safe_star.gaussian_sample(
            rng,
            self.get_loc(),
            self.get_scale(),
            n,
            max_iters,
            initialization_opt,
            stability_eps,
            self.get_input_bounds(),
        )
    }
}
