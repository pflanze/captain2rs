use captain2rs::biodivsim::sim_grid::dispersal_distances_threshold;

fn main() {
    let x = dispersal_distances_threshold(250, 0.1, 2);
    dbg!(x.sum());
}
