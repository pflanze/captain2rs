use captain2rs::biodivsim::empirical_grid::EmpiricalGrid;

fn main() -> anyhow::Result<()> {
    let mut grid = EmpiricalGrid::new(None, None);

    // Initialize from CSV with header: species,pu,amount
    grid.init_grid_from_puvsp_csv("puvsp.csv", None)?;

    // Optional: set disturbance (n_pus x 1)
    let n_pus = grid.protection_matrix().shape()[0];
    let disturbance = ndarray::Array2::<f64>::zeros((n_pus, 1));
    grid.set_disturbance_matrix(disturbance);

    // Subsample with seed for reproducibility
    grid.subsample_sp_h(Some(42))?;

    // Query some stats
    let species_totals = grid.individuals_per_species();
    println!("Totals per species: {:?}", species_totals);

    Ok(())
}
