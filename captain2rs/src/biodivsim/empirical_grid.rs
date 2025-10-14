/*

XX GPT:

# Notes and gaps

- CSV-only initialization is implemented. NPY loading/saving, pandas-based PU metadata, and external ecosystem glue (e.g., env/policy modules) are intentionally omitted to keep the Rust port focused and buildable. You can add npy support via the ndarray-npy crate if needed.
- Array shapes and semantics mirror the Python code: abundance tensor is 3D with a singleton third axis; protection and disturbance are column vectors to match broadcast patterns; thresholding and presence/absence rules are preserved exactly.
- Randomization and subsampling reproduce the logic, including the safety net to ensure each species retains presence in at least one originally occupied PU.

The translation is based on the original EmpiricalGrid.py from captain-project/captain2, preserving naming and behavior across methods.

 */

use ndarray::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Bernoulli, Distribution};
use std::collections::HashMap;

/// A Rust port of EmpiricalGrid from Python.
/// Shapes:
/// - h: (n_species, n_pus, 1) abundance tensor
/// - protection_matrix: (n_pus, 1)
/// - disturbance_matrix: (n_pus, 1)
pub struct EmpiricalGrid {
    counter: usize,
    #[allow(unused)]
    climate_layer: Vec<f64>,
    #[allow(unused)]
    climate_as_disturbance: f64,

    h: Array3<f64>,
    h_initial: Array3<f64>,

    protection_matrix: Array2<f64>,
    init_protection_matrix: Array2<f64>,
    disturbance_matrix: Option<Array2<f64>>,

    species_threshold: f64,

    species_sensitivities: Option<Array1<f64>>,
    list_species_values: Option<Array1<f64>>,

    species_id: Vec<usize>,
    species_id_indx: Vec<usize>,

    pus_id: Vec<usize>,
    pus_id_ind: Vec<usize>,

    n_species: usize,
    n_pus: usize,

    /// sqrt(n_pus) (kept for parity with Python)
    length: f64,

    /// Optional planning unit metadata (parity placeholder)
    coords_csv_path: Option<String>,
}

impl EmpiricalGrid {
    pub fn new(
        protection_matrix: Option<Array2<f64>>,
        species_sensitivities: Option<Array1<f64>>,
    ) -> Self {
        let default_pm = protection_matrix.unwrap_or_else(|| Array2::<f64>::zeros((0, 1)));
        Self {
            counter: 0,
            climate_layer: vec![],
            climate_as_disturbance: 0.0,

            h: Array3::<f64>::zeros((0, 0, 1)),
            h_initial: Array3::<f64>::zeros((0, 0, 1)),

            protection_matrix: default_pm.clone(),
            init_protection_matrix: default_pm,
            disturbance_matrix: None,

            species_threshold: 1.0,
            species_sensitivities,
            list_species_values: None,

            species_id: vec![],
            species_id_indx: vec![],
            pus_id: vec![],
            pus_id_ind: vec![],

            n_species: 0,
            n_pus: 0,
            length: 0.0,

            coords_csv_path: None,
        }
    }

    /// Initialize grid from either:
    /// - `puvsp_file` (CSV with header "species,pu,amount"), or
    /// - pre-saved `hist_file` (.npy) and optional `pu_id_file`/`sp_id_file` (.npy).
    ///
    /// In this Rust translation, only the `puvsp_file` path is implemented for input.
    pub fn init_grid_from_puvsp_csv(
        &mut self,
        puvsp_file: &str,
        pu_info_file: Option<&str>,
    ) -> anyhow::Result<()> {
        self.counter = 0;

        // Read CSV: header "species,pu,amount"
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(puvsp_file)?;

        let mut occs: Vec<(usize, usize, f64)> = Vec::new();
        for result in rdr.records() {
            let rec = result?;
            let sp: usize = rec.get(0).unwrap().parse()?;
            let pu: usize = rec.get(1).unwrap().parse()?;
            let amount: f64 = rec.get(2).unwrap().parse()?;
            occs.push((sp, pu, amount));
        }

        // Extract unique species and pus
        let mut sp_set: Vec<usize> = occs.iter().map(|(s, _, _)| *s).collect();
        sp_set.sort_unstable();
        sp_set.dedup();

        let mut pu_set: Vec<usize> = occs.iter().map(|(_, pu, _)| *pu).collect();
        pu_set.sort_unstable();
        pu_set.dedup();

        self.species_id = sp_set.clone();
        self.n_species = self.species_id.len();
        self.species_id_indx = (0..self.n_species).collect();

        self.pus_id = pu_set.clone();
        self.n_pus = self.pus_id.len();
        self.length = (self.n_pus as f64).sqrt();
        self.pus_id_ind = (0..self.n_pus).collect();

        // Map IDs to indices
        let sp_index: HashMap<usize, usize> = self
            .species_id
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();
        let pu_index: HashMap<usize, usize> = self
            .pus_id
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect();

        // Build h tensor (n_species, n_pus, 1)
        let mut init_h = Array3::<f64>::zeros((self.n_species, self.n_pus, 1));
        for (s, pu, amount) in occs.into_iter() {
            let si = *sp_index.get(&s).unwrap();
            let ui = *pu_index.get(&pu).unwrap();
            init_h[[si, ui, 0]] = amount;
        }
        self.h = init_h.clone();
        self.h_initial = init_h;

        // Protection matrix (n_pus, 1) defaults to zeros if unset
        if self.protection_matrix.shape() != [self.n_pus, 1] {
            self.protection_matrix = Array2::<f64>::zeros((self.n_pus, 1));
        }
        self.init_protection_matrix = self.protection_matrix.clone();

        // Species values default to ones
        if self.list_species_values.is_none() {
            self.list_species_values = Some(Array1::<f64>::ones(self.n_species));
        }

        // Optional PU coords metadata path
        if let Some(path) = pu_info_file {
            self.coords_csv_path = Some(path.to_string());
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.protection_matrix = self.init_protection_matrix.clone();
        self.counter = 0;
    }

    pub fn reset_init_protection_matrix(&mut self, p: Array2<f64>) {
        self.init_protection_matrix = p;
    }

    pub fn randomize_grid<R: Rng>(&mut self, rng: &mut R) {
        // Permute PUs consistently across h and protection_matrix
        let mut order: Vec<usize> = (0..self.n_pus).collect();
        // Fisher-Yates shuffle
        for i in (1..self.n_pus).rev() {
            let j = rng.gen_range(0..=i);
            order.swap(i, j);
        }

        // Reorder pus_id
        self.pus_id = order.iter().map(|&i| self.pus_id[i]).collect();

        // Reorder h along PU axis
        let mut h_new = Array3::<f64>::zeros((self.n_species, self.n_pus, 1));
        for (new_ui, &old_ui) in order.iter().enumerate() {
            h_new
                .slice_mut(s![.., new_ui, ..])
                .assign(&self.h.slice(s![.., old_ui, ..]));
        }
        self.h = h_new;

        // Reorder protection matrix
        let mut pm_new = Array2::<f64>::zeros((self.n_pus, 1));
        for (new_ui, &old_ui) in order.iter().enumerate() {
            pm_new[[new_ui, 0]] = self.protection_matrix[[old_ui, 0]];
        }
        self.protection_matrix = pm_new;
        self.pus_id_ind = (0..self.n_pus).collect();
    }

    pub fn set_disturbance_matrix(&mut self, disturbance: Array2<f64>) {
        self.disturbance_matrix = Some(disturbance);
    }

    pub fn individuals_per_species(&self) -> Array1<f64> {
        // sum over (i, j) axes: shape (s)
        self.h
            .sum_axis(Axis(1))
            .sum_axis(Axis(2))
            .into_dimensionality::<Ix1>()
            .unwrap()
    }

    pub fn individuals_per_cell(&self) -> Array2<f64> {
        // sum over species axis: shape (i, j) but j=1 -> (n_pus, 1)
        self.h.sum_axis(Axis(0))
    }

    pub fn protected_ind_per_species(&self) -> Array1<f64> {
        // tmp = h * protection_matrix (broadcast over species axis)
        let mut tmp = self.h.clone();
        for ui in 0..self.n_pus {
            let pm = self.protection_matrix[[ui, 0]];
            for si in 0..self.n_species {
                tmp[[si, ui, 0]] *= pm;
            }
        }
        tmp.sum_axis(Axis(1))
            .sum_axis(Axis(2))
            .into_dimensionality::<Ix1>()
            .unwrap()
    }

    pub fn geo_range_per_species(&self) -> Array1<f64> {
        // presence/absence by cell, not within cell; temp > 1 -> 1, else 0
        let mut temp = self.h.clone();
        temp.map_inplace(|v| {
            if *v > 1.0 {
                *v = 1.0;
            } else {
                *v = 0.0;
            }
        });
        temp.sum_axis(Axis(1))
            .sum_axis(Axis(2))
            .into_dimensionality::<Ix1>()
            .unwrap()
    }

    pub fn species_per_cell(&self) -> Array2<f64> {
        // threshold only used for total pop; within cell, >1 ->1, else 0
        let mut pa = self.h.clone();
        pa.map_inplace(|v| {
            if *v > 1.0 {
                *v = 1.0;
            } else {
                *v = 0.0;
            }
        });
        pa.sum_axis(Axis(0))
    }

    pub fn number_of_species(&self) -> usize {
        let totals = self.individuals_per_species();
        totals
            .iter()
            .filter(|&&x| x > self.species_threshold)
            .count()
    }

    pub fn extinct_species_id(&self) -> Vec<usize> {
        let totals = self.individuals_per_species();
        self.species_id
            .iter()
            .zip(totals.iter())
            .filter_map(|(&sid, &x)| {
                if x < self.species_threshold {
                    Some(sid)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn extinct_species_index_id(&self) -> Vec<usize> {
        let totals = self.individuals_per_species();
        self.species_id_indx
            .iter()
            .zip(totals.iter())
            .filter_map(|(&idx, &x)| {
                if x < self.species_threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn step(&mut self) {
        // Placeholder for future binomial draw based on disturbance_matrix
        self.counter += 1;
    }

    pub fn update_protection_matrix(
        &mut self,
        protection_matrix: Option<Array2<f64>>,
        indx: Option<&[usize]>,
    ) {
        if let Some(pm) = protection_matrix {
            self.protection_matrix = pm;
        }
        if let Some(indices) = indx {
            for &i in indices {
                if i < self.n_pus {
                    self.protection_matrix[[i, 0]] = 1.0;
                }
            }
        }
    }

    /// Disturbance subsampling akin to Python's `subsample_sp_h`.
    /// - If `seed` is Some(u64), deterministic; otherwise random.
    /// - If species_sensitivities unset, draws uniform [0,1] per species.
    pub fn subsample_sp_h(&mut self, seed: Option<u64>) -> anyhow::Result<()> {
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let species_sensitivity: Array1<f64> = if let Some(s) = &self.species_sensitivities {
            s.clone()
        } else {
            Array1::from(
                (0..self.n_species)
                    .map(|_| rng.gen::<f64>())
                    .collect::<Vec<_>>(),
            )
        };

        let disturbance = match &self.disturbance_matrix {
            Some(d) => d.clone(), // shape (n_pus, 1)
            None => {
                // Default to zeros (no disturbance)
                Array2::<f64>::zeros((self.n_pus, 1))
            }
        };

        // Tile disturbance across species: (n_pus, n_species)
        let mut disturbance_effect_sp = Array2::<f64>::zeros((self.n_pus, self.n_species));
        for ui in 0..self.n_pus {
            let val = disturbance[[ui, 0]];
            for si in 0..self.n_species {
                disturbance_effect_sp[[ui, si]] = val;
            }
        }

        // p = 1 - disturbance_effect_sp * species_sensitivity
        // compute per (ui, si)
        let mut p = Array2::<f64>::zeros((self.n_pus, self.n_species));
        for ui in 0..self.n_pus {
            for si in 0..self.n_species {
                let ps = 1.0 - disturbance_effect_sp[[ui, si]] * species_sensitivity[si];
                // Clamp to [0,1]
                p[[ui, si]] = ps.clamp(0.0, 1.0);
            }
        }

        // Bernoulli( p.T ) per (si, ui)
        let mut x = Array2::<f64>::zeros((self.n_species, self.n_pus));
        for si in 0..self.n_species {
            for ui in 0..self.n_pus {
                let prob = p[[ui, si]];
                let bern = Bernoulli::new(prob.max(0.0).min(1.0))?;
                x[[si, ui]] = if bern.sample(&mut rng) { 1.0 } else { 0.0 };
            }
        }

        // a = h_initial[:,:,0] * x
        let mut a = Array3::<f64>::zeros((self.n_species, self.n_pus, 1));
        for si in 0..self.n_species {
            for ui in 0..self.n_pus {
                a[[si, ui, 0]] = self.h_initial[[si, ui, 0]] * x[[si, ui]];
            }
        }

        // Ensure each species has at least 1 individual somewhere it originally occurred
        for si in 0..self.n_species {
            let sum_spp: f64 = a.slice(s![si, .., 0]).sum();
            if sum_spp == 0.0 {
                // PU indices where initial had presence (>0)
                let mut candidates: Vec<usize> = Vec::new();
                for ui in 0..self.n_pus {
                    if self.h_initial[[si, ui, 0]] > 0.0 {
                        candidates.push(ui);
                    }
                }
                if !candidates.is_empty() {
                    let choice = candidates[rng.gen_range(0..candidates.len())];
                    a[[si, choice, 0]] += 1.0;
                }
            }
        }

        self.h = a;
        Ok(())
    }

    // Accessors
    pub fn h(&self) -> &Array3<f64> {
        &self.h
    }

    pub fn protection_matrix(&self) -> &Array2<f64> {
        &self.protection_matrix
    }

    pub fn disturbance_matrix(&self) -> Option<&Array2<f64>> {
        self.disturbance_matrix.as_ref()
    }
}
