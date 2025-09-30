use std::{thread::sleep, time::Duration};

use captain2rs::biodivsim::sim_grid::dispersal_distances_threshold;

fn main() {
    let x = dispersal_distances_threshold(1000, 0.1, 3);
    dbg!(x.len());
    sleep(Duration::from_secs(10));
}
