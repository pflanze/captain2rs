use std::{thread::sleep, time::Duration};

use anyhow::Result;
use captain2rs::biodivsim::sim_grid_original::dispersal_distances_threshold;

fn main() -> Result<()> {
    let debug = true;
    if debug {
        let mut x = dispersal_distances_threshold(3, 0.1, 1, false, None);
        x.set([0, 1, 0, 2], 1.23456)?;
        assert_eq!(x[[0, 2, 3, 4]], 0.);
        dbg!(&x);
    }

    {
        let mut x = dispersal_distances_threshold(1000, 0.1, 3, true, Some(6.993));
        dbg!(x.len());
        assert_eq!(x.len(), 48832145);
        assert!(!x.is_sorted());
        x.sort()?;
        assert_eq!(x.get([999, 999, 999, 999]), Some(1.0));
        assert_eq!(x.get([0, 2, 4, 5]), None);
        assert_eq!(x[[0, 2, 4, 5]], 6.993);
        x.insert_unordered([999, 1000, 3, 4], 1.3);
        assert!(x.is_sorted());
        x.set([0, 2, 3, 4], 1.2)?;
        assert!(x.is_sorted());

        let slow = true;
        if slow {
            x.sort()?;
            assert!(x.is_sorted());
            assert_eq!(x.get([0, 2, 3, 4]), Some(1.2));
            assert_eq!(x[[0, 2, 3, 4]], 1.2);
            assert_eq!(x.get([0, 2, 3, 54]), None);
            assert_eq!(
                x.insert([0, 2, 3, 5], 1.2).err().unwrap().to_string(),
                "not strictly increasing coordinates: [999, 1000, 3, 4] vs. [0, 2, 3, 5]"
            );
            x.sort()?;
        }
        let memtop = false;
        if memtop {
            sleep(Duration::from_secs(10));
        }
        Ok(())
    }
}
