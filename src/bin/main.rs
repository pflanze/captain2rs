use std::{thread::sleep, time::Duration};

use anyhow::Result;
use captain2rs::biodivsim::sim_grid::dispersal_distances_threshold;

fn main() -> Result<()> {
    let mut x = dispersal_distances_threshold(1000, 0.1, 3);
    dbg!(x.len());
    x.sort()?;
    dbg!(x.len());
    // x.insert_unchecked((0, 2, 3, 4), 1.2);
    // x.insert_unchecked((0, 2, 3, 4), 1.2);
    dbg!(x.get((0, 2, 3, 4)));
    x.insert_unchecked((0, 2, 3, 4), 1.2);
    x.sort()?;
    dbg!(x.get((0, 2, 3, 4)));
    dbg!(x.get((0, 2, 3, 54)));

    // x.insert((0, 2, 3, 5), 1.2)?;
    // x.sort()?;
    // sleep(Duration::from_secs(10));
    Ok(())
}
