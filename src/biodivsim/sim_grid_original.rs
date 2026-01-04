use ndarray::ArrayView2;

use crate::{
    biodivsim::div::{Float, bounded_range_around},
    coo::Coo,
};

// @jit(nopython=True)
// def dispersalDistancesThreshold(length: int,
//                                 lambda_0: float,
//                                 threshold=3):
//     print("calculating distances with threshold...")
//     dumping_dist = np.zeros((length, length, length, length))
//     for i in range(0, length):
//         for j in range(0, length):
//             for n in range(max([0, i-threshold]), min([length, i+threshold])):
//                 for m in range(max([0, j-threshold]), min([length, j+threshold])):
//                     exp_rate = 1.0 / lambda_0
//                     # relative dispersal probability: always 1 at distance = 0
//                     # the actual number of offspring is modulated by growth_rate
//                     dumping_dist[i, j, n, m] = np.exp(
//                         -exp_rate * np.sqrt((i - n) ** 2 + (j - m) ** 2)
//                     )
//     return dumping_dist

/// Compute dispersal distances with a threshold.
///
/// # Arguments
/// * `length` - size of the grid (length x length)
/// * `lambda_0` - dispersal parameter
/// * `threshold` - neighborhood radius
///
/// # Returns
/// A 4D array of shape (length, length, length, length)
pub fn dispersal_distances_threshold(
    length: u32,
    lambda_0: Float,
    threshold: u32,
    test_hack: bool,
    default: Option<Float>,
) -> Coo<i64, 4, Float> {
    println!("calculating distances with threshold...");

    let mut dumping_dist = Coo::new(default.unwrap_or(0.));
    let exp_rate = 1.0 / lambda_0;

    for i in 0..length {
        for j in 0..length {
            for n in bounded_range_around(i, length, threshold) {
                for m in bounded_range_around(j, length, threshold) {
                    let dx = (i as Float - n as Float).powi(2); // XX why not in i64!
                    let dy = (j as Float - m as Float).powi(2);
                    let dist = (dx + dy).sqrt();
                    dumping_dist
                        .insert([i, j, n, m], (-exp_rate * dist).exp())
                        .expect("inserts to happen in sorted order");
                }
            }
        }
    }

    if test_hack {
        dumping_dist.insert_unordered([5, 9, 7, 1], -7.3456);
    }

    dumping_dist
}

/// Compute dispersal distances using geographic coordinates with a threshold.
///
/// # Arguments
/// * `length` - size of the grid (length x length)
/// * `lambda_0` - dispersal parameter
/// * `lat` - 2D array of latitude coordinates (length x length)
/// * `lon` - 2D array of longitude coordinates (length x length)
/// * `threshold` - neighborhood radius (coordinate units)
///
/// # Returns
/// A 4D Coo sparse array of shape (length, length, length, length)
pub fn dispersal_distances_coord(
    length: u32,
    lambda_0: Float,
    lat: ArrayView2<Float>,
    lon: ArrayView2<Float>,
    threshold: Float,
) -> Coo<i64, 4, Float> {
    assert_eq!(lat.shape(), [length as usize, length as usize]);
    assert_eq!(lon.shape(), [length as usize, length as usize]);

    println!("calculating distances with coordinate threshold...");

    let mut dumping_dist = Coo::new(0.0);
    let exp_rate = 1.0 / lambda_0;

    let len_usize = length as usize;

    for i in 0..len_usize {
        for j in 0..len_usize {
            let lat_ij = lat[(i, j)];
            let lon_ij = lon[(i, j)];

            for n in 0..len_usize {
                for m in 0..len_usize {
                    let lat_nm = lat[(n, m)];
                    let lon_nm = lon[(n, m)];

                    let lat_diff = (lat_ij - lat_nm).abs();
                    let lon_diff = (lon_ij - lon_nm).abs();

                    if lat_diff <= threshold && lon_diff <= threshold {
                        // Calculate Euclidean distance in coordinate space
                        let dx = (lat_ij - lat_nm).powi(2);
                        let dy = (lon_ij - lon_nm).powi(2);
                        let dist = (dx + dy).sqrt();

                        // Calculate dispersal probability and insert into the sparse matrix
                        dumping_dist
                            .insert(
                                [i as i64, j as i64, n as i64, m as i64],
                                (-exp_rate * dist).exp(),
                            )
                            .expect("inserts to happen in sorted order");
                    }
                }
            }
        }
    }

    dumping_dist
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[cfg(debug_assertions)]
    const RELEASE: bool = false;
    #[cfg(not(debug_assertions))]
    const RELEASE: bool = true;

    #[test]
    fn tests() -> Result<()> {
        let debug = true;
        if debug {
            let mut x = dispersal_distances_threshold(3, 0.1, 1, false, None);
            x.set([0, 1, 0, 2], 1.23456)?;
            assert_eq!(x[[0, 2, 3, 4]], 0.);
            dbg!(&x);
        }

        let run_slow_tests = RELEASE;
        if run_slow_tests {
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

            let memtop = false;
            if memtop {
                use std::{thread::sleep, time::Duration};
                sleep(Duration::from_secs(10));
            }
        }
        Ok(())
    }
}
