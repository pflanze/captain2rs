use std::{thread::sleep, time::Duration};

use anyhow::Result;
use captain2rs::biodivsim::sim_grid::dispersal_distances_threshold;

fn main() -> Result<()> {
    let mut x = dispersal_distances_threshold(1000, 0.1, 3);
    dbg!(x.len());
    assert_eq!(x.len(), 35892081);
    assert!(x.is_sorted());
    assert_eq!(x.get([999, 999, 999, 999]), Some(1.0));
    assert_eq!(x.get([0, 2, 3, 4]), None);
    x.insert_unchecked([999, 1000, 3, 4], 1.3);
    assert!(x.is_sorted());
    x.insert_unchecked([0, 2, 3, 4], 1.2);
    assert!(!x.is_sorted());

    let slow = false;
    if slow {
        x.sort()?;
        assert!(x.is_sorted());
        assert_eq!(x.get([0, 2, 3, 4]), Some(1.2));
        assert_eq!(x.get([0, 2, 3, 54]), None);
        assert_eq!(
            x.insert([0, 2, 3, 5], 1.2).err().unwrap().to_string(),
            "not strictly increasing coordinates: [999, 999, 999, 999] vs. [0, 2, 3, 5]"
        );
        x.sort()?;
    }
    let memtop = false;
    if memtop {
        sleep(Duration::from_secs(10));
    }
    Ok(())
}
