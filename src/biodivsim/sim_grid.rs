//! New correspondent file to `captain/biodivsim/SimGrid.py` (for the
//! old direct translations, see
//! [sim_grid_original.rs](sim_grid_original.rs).

use std::{collections::HashMap, ops::Range, sync::Arc};

use anyhow::bail;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::iter::ParallelIterator;

use crate::{
    biodivsim::{
        div::{Float, RealFloat},
        sparse_dispersal::SparseDispersal,
    },
    debug, def_id,
    evaluation_cache::EvalForCache,
    id_array::{IdArray1, IdArray3},
    id_vec::IdVec,
    sparse::{Sparse2, SparseMask},
    utillib::arc::CloneArc,
    warn,
};

pub trait PickleInitializer {
    fn get_initial_state(&self, a: ArrayView2<Float>, n: usize, len: usize) -> Array3<Float>;
}

// #[derive(Debug)]
// struct UnknownPickleInitializer;

def_id! {pub OrganismId}

#[derive(Debug)]
struct Unknown;

/// XX?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClimateModelKind(u8);

pub struct ClimateModel;

impl ClimateModel {
    pub fn new(kind: ClimateModelKind) -> Self {
        todo!()
    }
    pub fn update_climate(&mut self, a: Array2<Float>) -> Array2<Float> {
        todo!()
    }
}

pub enum GrowthRate {
    // Vector of the same length of species
    BelowNumSpecies(Array1<Float>),
    FromNumSpecies(Float),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DispersalWhen {
    BeforeDeath,
    AfterDeath,
}

impl TryFrom<u8> for DispersalWhen {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        // "set to 1/0 to get dispersing pool before/after death"? XX not sure
        match value {
            1 => Ok(DispersalWhen::BeforeDeath),
            0 => Ok(DispersalWhen::AfterDeath),
            _ => bail!("invalid value for DispersalWhen, allowed are 0 and 1"),
        }
    }
}

pub struct SimGridParamsWithoutDefaults {
    length: usize,
    num_species: usize,
    species_ids: Range<usize>,
    alpha: Float,
    /// initial (max) carrying capacity
    k_max: Float,
    dispersal_parameters: IdVec<OrganismId, DispersalParameters>,
    disturbance_initializer: Float, // XX?
    /// vector of sensitivity per species
    disturbance_sensitivity: Array1<Float>, // XX? XX also,  species as a key type wrap?
    rnd_alpha_species: Float,       // XX?
    truncate_to_int: bool,
    disturbance_matrix_diff: Float,
}

pub struct SimGridParamsWithDefaults {
    selectivedisturbance_initializer: Float, // XX?
    /// vector of selective sensitivity per species
    selective_sensitivity: Array1<Float>, // XX?
    immediate_capacity: bool,
    truncate_to_int: bool,
    species_threshold: Float,
    rnd_alpha: Float,           // XX?
    k_disturbance_coeff: Float, // XX?
    /// (cj:) taken out and moved to
    /// SimGrids.{selective_disturbance_matrix, protection_matrix} in
    /// SimGrid::new
    actions: Option<(Array2<Float>, Array2<Float>)>,
    /// "set to 1/0 to get dispersing pool before/after death", default 0
    dispersal_before_death: DispersalWhen,
    rnd_alpha_species: Float, // XX?
    climate_model: ClimateModelKind,
    growth_rate: GrowthRate,
    // phyloGenerator:
    climate_sensitivity: Array1<Float>, // XX?
    climate_as_disturbance: bool,       // XX? Python uses it as a boolean, but initialized to 1
    /// Unimplemented (in parts?)
    disturbance_dep_dispersal: bool, // XX? Python uses it as a boolean, but initialized to 1
    species_cell_specific_capacity: Option<Unknown>, // XX?

    /// How suitable a pixel is for a given species
    habitat_suitability: Option<IdArray3<OrganismId, Float>>,

    // (Daniele: not used here?)
    future_habitat_suitability: Option<Unknown>,

    /// Climate change: multiplier per step(?) for habitat_suitability
    delta_suitability_per_step: Option<IdArray3<OrganismId, Float>>,

    /// Individuals (but float), to detect/decide whether a species is present. Probably 1.
    species_threshold_per_cell: Float,

    /// max number of individuals of a species per cell -- used!
    k_species: Option<IdArray1<OrganismId, Float>>,

    rm_lingering_pops: bool,

    /// The names of species
    species_ids: Option<IdVec<OrganismId, String>>,
}

impl Default for SimGridParamsWithDefaults {
    fn default() -> Self {
        Self {
            selectivedisturbance_initializer: 0.,
            selective_sensitivity: Array1::from_vec(vec![]), // XX per species, must be num_species long??
            immediate_capacity: false,
            truncate_to_int: false,
            species_threshold: 1.,
            rnd_alpha: 1.,
            k_disturbance_coeff: 1.,
            actions: None,
            dispersal_before_death: DispersalWhen::try_from(0).unwrap(),
            rnd_alpha_species: 0.,
            climate_model: ClimateModelKind(0),
            growth_rate: GrowthRate::BelowNumSpecies(Array1::ones(1)),
            climate_sensitivity: Array1::from_vec(vec![]),
            climate_as_disturbance: true, // XX Python initializes it to 1
            disturbance_dep_dispersal: true, // XX Python initializes it to 1
            species_cell_specific_capacity: None,
            habitat_suitability: None,
            future_habitat_suitability: None,
            delta_suitability_per_step: None,
            species_threshold_per_cell: 1.,
            k_species: None,
            rm_lingering_pops: false,
            species_ids: None,
        }
    }
}

impl SimGridParamsWithDefaults {
    /// `k_species` reshaped as 3D array ("K_species[:, np.newaxis,
    /// np.newaxis] * habitat_suitability")
    pub fn k_species3d(&self) -> Option<ArrayView3<'_, Float>> {
        self.k_species.as_ref().map(|k_species| {
            k_species
                .view()
                .into_shape((k_species.len(), 1, 1))
                .expect("compatible statically")
        })
    }
}

/// Derived but constant values
struct SimGridDerived {
    disturbance_matrix: Array2<Float>,
    k_cells: Array2<Float>,
    species_carbon_value: Array1<Float>, // XX?
    reference_grid_pu: Option<Unknown>,
    n_pus: Option<Unknown>,
    alpha_histogram: Array3<Float>,
    climate_model: ClimateModel,
    climate_layer: Array2<Float>,
    selective_alpha_histogram: Array3<Float>,
    disturbance_effect_multiplier: Float, // XX?
    additional_carbon_matrix: Array2<Float>,
    selective_disturbance_matrix: Array2<Float>,
    protection_matrix: Array2<Float>,
}

/// (cj:) Parameters for SparseDispersal::new
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DispersalParameters {
    lambda_0: RealFloat,
}

impl EvalForCache<SparseDispersal, Arc<SparseMask>> for DispersalParameters {
    fn eval(&self, mask: &Arc<SparseMask>) -> SparseDispersal {
        let Self { lambda_0 } = self;
        let threshold = 3; // XX todo: calculate from lambda_0
        SparseDispersal::new(*lambda_0, threshold, mask.clone_arc())
    }
}

/// Initialized via `SimGrid::init_grid`
struct Grid {
    // was:
    // h: Array3<Float>,
    // now:
    /// Organism concentrations for each organism
    h: IdVec<OrganismId, Sparse2<Float>>,

    // Was:
    // dumping_dist: Array4<Float>,
    // now:
    /// A pre-generated cache of `SparseDispersal` datastructures for
    /// all `DispersalParameters` as determined by the organisms.
    dispersal_for: HashMap<DispersalParameters, SparseDispersal>,
}

/// Mutated values
struct SimGridState {
    /// (max) carrying capacity -- same for all organisms.
    k_max: Array2<Float>,
    counter: usize,
}

pub struct SimGrid {
    params_without_defaults: SimGridParamsWithoutDefaults,
    params_with_defaults: SimGridParamsWithDefaults,
    derived: SimGridDerived,
    grid: Option<Grid>,
    state: SimGridState,
}

// #[test]
// fn t_simgrid_size() {
//     assert_eq!(size_of::<SimGridParamsWithoutDefaults>(), 192); // not 96
//     assert_eq!(size_of::<SimGridParamsWithDefaults>(), 384); // not 160
//     assert_eq!(size_of::<SimGridDerived>(), 608); // not 104
//     assert_eq!(size_of::<Grid>(), 144); // not 56. 144 = 18 words
//     assert_eq!(size_of::<SimGridState>(), 8);
//     assert_eq!(size_of::<SimGrid>(), 1336);
//     assert_eq!(size_of::<Array3<Float>>(), 80); // not 8 or 24
//     assert_eq!(size_of::<Array2<Float>>(), 64);
//     assert_eq!(size_of::<Array1<Float>>(), 48);
//     assert_eq!(size_of::<Vec<Float>>(), 24);
// }

pub struct StepOptions {
    action: Option<Unknown>,
    fast_dist: bool,
    skip_dispersal: bool,
    update_suitability: bool,
}

impl SimGrid {
    pub fn new(
        params_without_defaults: SimGridParamsWithoutDefaults,
        mut params_with_defaults: SimGridParamsWithDefaults,
    ) -> Self {
        let length = params_without_defaults.length;

        let disturbance_matrix = Array2::zeros((length, length));
        let k_cells = (1. - &disturbance_matrix) * params_without_defaults.k_max;

        let alpha_histogram = calculate_alpha_histogram(
            params_without_defaults.disturbance_sensitivity.view(),
            disturbance_matrix.view(),
        );

        let (selective_disturbance_matrix, protection_matrix) =
            if let Some(tuple) = params_with_defaults.actions.take() {
                tuple
            } else {
                (
                    Array2::zeros((length, length)),
                    Array2::zeros((length, length)),
                )
            };

        let selective_alpha_histogram = calculate_alpha_histogram(
            params_with_defaults.selective_sensitivity.view(),
            selective_disturbance_matrix.view(),
        );

        let mut climate_model = ClimateModel::new(params_with_defaults.climate_model);
        let climate_layer = if params_with_defaults.climate_model == ClimateModelKind(0) {
            Array2::zeros((length, length))
        } else {
            if params_with_defaults.climate_as_disturbance {
                climate_model.update_climate(Array2::zeros((length, length)))
            } else {
                climate_model.update_climate(Array2::ones((length, length)))
            }
        };

        // XX  if phyloGenerator == 0 ....

        // let k_species_cells = params_with_defaults.species_cell_specific_capacity.clone();
        let species_carbon_value = Array1::ones(params_without_defaults.num_species);
        // >1 to fast-forward effect of disturbance
        let disturbance_effect_multiplier = 1.;
        let additional_carbon_matrix = Array2::zeros((length, length));
        // let dumping_dist = params_without_defaults.precomputed_dispersal_probs.clone();

        let reference_grid_pu = None;
        let n_pus = None;

        let k_max = params_without_defaults.k_max * Array2::ones((length, length));

        Self {
            params_without_defaults,
            params_with_defaults,
            derived: SimGridDerived {
                disturbance_matrix,
                k_cells,
                species_carbon_value,
                reference_grid_pu,
                n_pus,
                alpha_histogram,
                climate_model,
                climate_layer,
                selective_alpha_histogram,
                disturbance_effect_multiplier,
                additional_carbon_matrix,
                selective_disturbance_matrix,
                protection_matrix,
            },
            grid: None,
            state: SimGridState { k_max, counter: 0 },
        }
    }

    fn doing_dispersal_before_death(&self) -> bool {
        match self.params_with_defaults.dispersal_before_death {
            DispersalWhen::BeforeDeath => true,
            DispersalWhen::AfterDeath => false,
        }
    }

    /// XX initialize with random data? todo: rename method accordingly?
    pub fn init_grid(&mut self, state_initializer: impl PickleInitializer) {
        // XX println!(
        //     "\nself._dumping_dist {:?}",
        //     self.grid.as_ref().map(|v| &v.dumping_dist)
        // );

        // random histogram
        let h = state_initializer.get_initial_state(
            self.state.k_max.view(),
            self.params_without_defaults.num_species,
            self.params_without_defaults.length,
        );
        // init dumping factors (unless already provided)
        if let Some(grid) = &mut self.grid {
            grid.h = sparse_h_from_array3(h);
        } else {
            todo!()
            // self.grid = Some(Grid {
            //     h,
            //     dumping_dist: SparseDispersal::new(
            //         self.params_without_defaults.lambda_0,
            //         3, // XX?
            //         mask.clone_arc(),
            //     ),
            // });
        }

        // XX self.updateAlphaHistogram(); //  (Design XX?)

        // XX this is not used anywhere, is it?
        // self._climate_opt_sp_3D, self._climate_range_sp_3D = self.getClimateTolerance()

        if self.params_with_defaults.disturbance_dep_dispersal {
            unimplemented!("Disturbance-dependent dispersal not implemented");
            // self._diag_list = getDiag.get_diagonals_from_pickle("../scripts/diagonals50.pkl")
        }
    }

    pub fn step(&mut self, options: &StepOptions) {
        let StepOptions {
            action,
            fast_dist,
            skip_dispersal,
            update_suitability,
        } = options;
        debug!("getting NumCandidates");

        // let num_candidates;
        // let norm_candidates;
        if !self.doing_dispersal_before_death() {
            // np.einsum("sij,ijnm->snm", self._h, self._dumping_dist)
            // let num_candidates =
            if let Some(grid) = &mut self.grid {
                grid.h.par_iter_mut_enumerated().for_each(|(id, h)| {
                    let params = &self.params_without_defaults.dispersal_parameters[id];
                    grid.dispersal_for[params]
                        .apply_mut(h, true)
                        .expect("the masks match")
                });
            } else {
                warn!("have no grid!");
            }
        }
        todo!();

        self.state.counter += 1;
    }
}

/// Given `disturbanceSensitivity` of shape (S,) and
/// `disturbanceMatrix` of shape (I, J), returns shape (S, I, J).  The
/// result is essentially a stack of S copies of `disturbanceMatrix`;
/// each copy scaled by the corresponding element of
/// `disturbanceSensitivity`.  XX optimize: avoid materialization?!
fn calculate_alpha_histogram(
    disturbance_sensitivity: ArrayView1<Float>,
    disturbance_matrix: ArrayView2<Float>,
) -> Array3<Float> {
    // # TODO: implement 3D disturbance
    // # print("disturbanceSensitivity.shape", disturbanceSensitivity)
    // # if len(disturbanceSensitivity.shape) == 2:
    // #     "when alphaHistogram==0: nobody dies, when==1: all die"
    // #     return np.einsum("sd,dij->sij", disturbanceSensitivity, disturbanceMatrix)
    // # if len(disturbanceSensitivity.shape) == 1:
    // "when alphaHistogram==0: nobody dies, when==1: all die"
    // return np.einsum("s,ij->sij", disturbanceSensitivity, disturbanceMatrix)
    let s = disturbance_sensitivity.dim();
    let (i, j) = disturbance_matrix.dim();
    let disturbance_sensitivity = disturbance_sensitivity
        .insert_axis(Axis(1))
        .insert_axis(Axis(2));
    let disturbance_matrix = disturbance_matrix.insert_axis(Axis(0));
    let r = disturbance_sensitivity.to_owned() * disturbance_matrix;
    debug_assert_eq!((s, i, j), r.dim());
    r
}

#[test]
fn t_calculate_alpha_histogram() {
    use numpy::array;
    let sensitivity = array![2., 0.5];
    let matrix = array![[0.1, 1., 0.5], [0.0, 0.4, 0.6]];
    let r = calculate_alpha_histogram(sensitivity.view(), matrix.view());
    let expected = array![
        [[0.2, 2.0, 1.], [0.0, 0.8, 1.2]],
        [[0.05, 0.5, 0.25], [0.0, 0.2, 0.3]]
    ];
    assert_eq!(&r, &expected);
}

fn sparse_h_from_array3(h: Array3<Float>) -> IdVec<OrganismId, Sparse2<Float>> {
    todo!("hmm where are coords?")
}
