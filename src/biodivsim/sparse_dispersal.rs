//! Dispersal on sparse matrices (maps). Organisms cannot go into the
//! sparse region (although they could jump over a sparse region if
//! thin enough).

use std::{mem::swap, sync::Arc};

use ndarray::ArrayView2;

use crate::{
    _dispersal_dispatch,
    biodivsim::{
        dispersal::{Dispersal, len_from_threshold},
        div::{FlippedBoundedRangesAround, Float, RealFloat, bounded_range_around_w_clipped},
    },
    perhaps_dump,
    sparse::{Sparse, SparseFromViewAndMaskError, SparseMask},
    utillib::arc::CloneArc,
};

#[derive(Debug)]
pub struct SparseDispersal {
    dispersal: Dispersal,
    weights: Sparse<Float>,
}

#[derive(thiserror::Error, Debug)]
pub enum SparseDispersalApplyError {
    #[error("the data passed to the dispersal does not match its mask (pointer equality check)")]
    NotEqMask,
}

impl SparseDispersal {
    /// Carries the cost of creating the threshold cache and
    /// especially weights matrix of the same size as the image, as
    /// well as its compression.
    pub fn new(lambda_0: RealFloat, threshold: usize, mask: Arc<SparseMask>) -> Self {
        let dispersal = Dispersal::new(lambda_0.into(), threshold);

        // The weight at each point comes from calculating the
        // convolution over 0 (for sparse points) and 1 (for
        // inhabitable points) values.
        let oneszeroes = mask.clone_arc().to_ones_and_zeros();

        let weights = dispersal.apply(oneszeroes.view());
        if false {
            // dbg!(&mask);
            dbg!(&dispersal.cache);
            dbg!(&oneszeroes);
            dbg!(&weights);
        }
        perhaps_dump!(
            "SparseDispersal-new_dispersal-cache",
            dispersal.cache.view(),
            0. ..1.
        );
        perhaps_dump!("SparseDispersal-new_weights", weights.view(), 0. ..10.);

        let weights = Sparse::from_view_and_mask(weights.view(), mask).unwrap_or_else(
            |SparseFromViewAndMaskError::DimensionsDoNotMatch| {
                unreachable!("we carried over dimensions so they do match")
            },
        );
        Self { dispersal, weights }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.weights.dim()
    }

    pub fn width(&self) -> usize {
        self.weights.width()
    }

    pub fn height(&self) -> usize {
        self.weights.height()
    }

    pub fn mask(&self) -> &Arc<SparseMask> {
        self.weights.mask()
    }

    pub fn cache(&self) -> ArrayView2<'_, Float> {
        self.dispersal.cache.view()
    }

    pub fn weights(&self) -> &Sparse<Float> {
        &self.weights
    }

    fn _apply_mut<const M: usize>(&self, a: &mut Sparse<Float>, equalize: bool) {
        let old_sum = if equalize { Some(a.sum()) } else { None };

        let (height, width) = self.dim();

        let threshold = self.dispersal.threshold;
        let threshold_len = len_from_threshold(threshold);

        // Sliding window of rows, to try to keep the data within the
        // CPU cache. Will shift the inner `Vec`s to reuse their heap
        // memory to avoid spilling to main memory).
        let mut rows: Vec<Vec<Float>> = Vec::with_capacity(height);
        // Create the pool of internal `Vec`s--fill in the max number
        // we need, which is more rows than needed at first--`if
        // n_is_unclipped` below takes care of that.
        {
            for row_i in 0..threshold_len.min(height) {
                let mut vec = Vec::with_capacity(width);
                vec.resize(width, 0.);
                rows.push(vec);
                a.mut_slice_row(row_i, 0., &mut rows[row_i]);
            }
        }

        for row_i in 0..height {
            let mut weights = self.weights.compressed_row(row_i).iter();
            // Only calculate points which are non-null: iterate over
            // those only
            for (x0, slice) in a.row_iter_mut(row_i) {
                for (i, rf) in slice.iter_mut().enumerate() {
                    let x = x0 as usize + i;

                    let area = FlippedBoundedRangesAround::new(row_i, height, x, width, threshold);
                    // dbg!(row_i, &area, &rows);
                    let sum = self.dispersal.dot::<M>(area, rows.as_slice());
                    let weight = weights.next().expect("fits because we checked eq masks");
                    *rf = sum / weight;
                }
            }

            assert!(weights.next().is_none());

            // Slide the window if necessary
            {
                let (n_range, n_is_unclipped) =
                    bounded_range_around_w_clipped(row_i, height, threshold);
                if n_is_unclipped && n_range.end < height {
                    // Need to move the first row of the window,
                    // replace with an empty Vec (which has no heap
                    // allocation).

                    // eprintln!(
                    //     "{n_range:?} n_is_unclipped={n_is_unclipped} \
                    //      -> moving row {} out to the end",
                    //     n_range.start
                    // );
                    let mut row = Vec::new();
                    swap(&mut row, &mut rows[n_range.start]);
                    a.mut_slice_row(n_range.end, 0., &mut row);
                    rows.push(row);
                }
            }
        }

        if let Some(old_sum) = old_sum {
            let new_sum = a.sum();
            *a *= old_sum / new_sum;
        }
    }

    /// Mutates `a` in place (this is possible because the algorithm
    /// uncompresses the rows within the threshold window before
    /// overwriting the values). Returns an error if `a` doesn't use
    /// the same mask. If `equalize` is true, then a (relatively
    /// cheap) post-processing step is applied to ensure that the sum
    /// over all values stays the same (you likely want this, unless
    /// you choose a small lambda and use small time steps).
    pub fn apply_mut(
        &self,
        a: &mut Sparse<Float>,
        equalize: bool,
    ) -> Result<(), SparseDispersalApplyError> {
        let am: &SparseMask = a.mask();
        let sm: &SparseMask = self.mask();
        if !std::ptr::eq(am, sm) {
            return Err(SparseDispersalApplyError::NotEqMask);
        }
        Ok(_dispersal_dispatch!(
            self.dispersal.threshold, { self._apply_mut::< } { >(a, equalize) }
        ))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ndarray::Array2;
    use numpy::array;

    use crate::{assert_eq_float, utillib::arc::IntoArc};

    use super::*;

    #[test]
    fn t_1() -> Result<()> {
        let mask_bits = array![
            [false, false, true, true],
            [false, true, true, false],
            [false, true, true, true],
            [false, true, true, true],
            [false, true, false, false],
            [true, false, false, false],
        ];
        let mask = SparseMask::from_bool_mask(mask_bits.view())?.into_arc();
        assert_eq!(mask.width(), 4);
        assert_eq!(mask.height(), 6);
        assert_eq!(mask.total_count_data(), 12);

        let dispersal = SparseDispersal::new(2.3.try_into().unwrap(), 1, mask.clone_arc());
        if false {
            dbg!(dispersal.cache());
            let weights = dispersal.weights().decompress(0.);
            dbg!(weights);
        }

        let data = array![
            [0., 0., 3., 3.4],
            [0., 2., 3.1, 0.],
            [0., 2.3, 2.1, 0.1],
            [0., 1.3, 1.0, 0.3],
            [0., 0.3, 0., 0.],
            [0.9, 0., 0., 0.],
        ];

        {
            let mut c = Sparse::from_view_and_pred(data.view(), |x| x == 0.)?;
            assert_eq!(&mask, c.mask());
            match dispersal.apply_mut(&mut c, true) {
                Ok(_) => panic!(
                    "expecting to get a NotEqMask error \
                     (since it is reconstructed, not pointer eq)"
                ),
                Err(SparseDispersalApplyError::NotEqMask) => (),
            }
        }

        let mut c = Sparse::from_view_and_mask(data.view(), mask.clone_arc())?;
        assert_eq_float!(data.sum(), c.decompress(0.).sum());

        dispersal.apply_mut(&mut c, true)?;

        let d = c.decompress(0.);

        // dbg!(&d);

        // Sum doesn't matter whether compressed or not.
        assert_eq_float!(c.data().sum(), d.sum());

        // Organisms have just moved, so the sum should be the same
        // before and after (well, we pass `true` above to enforce
        // this)

        assert_eq_float!(data.sum(), d.sum());

        // panic!();

        Ok(())
    }

    #[test]
    fn t_2() -> Result<()> {
        let threshold = 8;
        let lambda_0 = 0.3;
        let mask_bits = array![
            [true, true, false],
            [true, true, true],
            [true, true, true],
            [true, true, true],
        ];
        let mask = SparseMask::from_bool_mask(mask_bits.view())?.into_arc();
        assert_eq!(mask.width(), 3);
        assert_eq!(mask.height(), 4);
        assert_eq!(mask.total_count_data(), 11);

        let dispersal =
            SparseDispersal::new(lambda_0.try_into().unwrap(), threshold, mask.clone_arc());
        dbg!(dispersal.cache());
        let weights = dispersal.weights().decompress(0.);
        dbg!(weights);

        let mut data = Array2::<Float>::ones((4, 3)) * 4.;
        data[(3, 0)] = 0.5;

        let c0 = Sparse::from_view_and_mask(data.view(), mask.clone_arc())?;
        let mut c = c0.clone();
        let sum_before = c.decompress(0.).sum();
        // assert_eq_float!(data.sum() - 4., sum_before);//
        assert_eq_float!(c.sum(), sum_before);

        for _ in 0..10 {
            dispersal.apply_mut(&mut c, false)?;
        }

        dbg!(c.decompress(0.));

        // Organisms have just moved, so the sum should be the same
        // before and after, but since we're passing it `false` above,
        // they don't match precisely.
        // dbg!(sum_before, c.sum());
        assert_eq_float!(sum_before, 40.5);
        assert_eq_float!(c.sum(), 40.57559683532406);

        let mut c = c0.clone();
        for _ in 0..10 {
            dispersal.apply_mut(&mut c, true)?;
        }

        assert_eq_float!(sum_before, c.sum());

        // panic!();

        Ok(())
    }
}
