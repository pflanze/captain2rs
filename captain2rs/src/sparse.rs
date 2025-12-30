//! Sparse 2D arrays, based on a run-length encoded shareable mask
//! (strides of gap and data lengths). This works well for cases with
//! large run lengths, e.g. maps.

use std::{
    fmt::Debug,
    mem::{transmute, MaybeUninit},
    sync::Arc,
};

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Zero;

use crate::clone_arc;

/// A length of a gap or strip of points in one row of a matrix,
/// i.e. must be able to represent the max width of a matrix, and also
/// the max height (see `CountTotal`)
type Count = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stride {
    count_empty: Count,
    count_data: Count,
}

/// The sum of all counts across a matrix
type TotalCount = u32;

#[test]
fn assert_dependency() {
    assert_eq!(size_of::<TotalCount>(), 2 * size_of::<Count>());
}

enum StrideAction {
    Noop,
    Push(Stride),
}

impl Stride {
    pub fn is_empty(self) -> bool {
        let Stride {
            count_empty,
            count_data,
        } = self;
        count_empty.is_zero() && count_data.is_zero()
    }

    fn new() -> Stride {
        Stride {
            count_empty: 0,
            count_data: 0,
        }
    }

    /// Must remain private, would overflow if we didn't check that
    /// width <= Count::MAX
    fn add_null(&mut self) -> StrideAction {
        if self.count_data.is_zero() {
            // Can't overflow since we checked width <= Count::MAX
            self.count_empty += 1;
            StrideAction::Noop
        } else {
            let stride_to_push = *self;
            *self = Stride {
                count_empty: 1,
                count_data: 0,
            };
            StrideAction::Push(stride_to_push)
        }
    }
}

#[derive(Debug)]
struct StridesRowIndex {
    strides_start_i: TotalCount,
    strides_len: TotalCount,
    data_start_i: usize,
}

/// The information about where the values in a sparse array are.
#[derive(Debug)]
pub struct SparseMask {
    shape: [usize; 2],
    strides: Box<[Stride]>,
    /// To get the slices of `strides` and `data`
    row_index: Box<[StridesRowIndex]>,
    /// The number of non-sparse data points (i.e. the length of the
    /// `data` array in `Sparse`)
    total_count_data: TotalCount,
}

impl PartialEq for SparseMask {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.strides == other.strides
    }
}

impl Eq for SparseMask {}

impl SparseMask {
    pub fn total_count_data(&self) -> TotalCount {
        self.total_count_data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim(&self) -> (usize, usize) {
        let [height, width] = self.shape;
        (height, width)
    }

    pub fn width(&self) -> usize {
        let [_height, width] = self.shape;
        width
    }

    pub fn height(&self) -> usize {
        let [height, _width] = self.shape;
        height
    }

    pub fn row_len(&self) -> usize {
        self.width()
    }

    pub fn num_values(&self) -> usize {
        let (height, width) = self.dim();
        height * width
    }

    /// How many bytes of data this instance uses (malloc overheads not included)
    pub fn stats_bytes(&self) -> usize {
        size_of::<Self>()
            + self.strides.len() * size_of::<Stride>()
            + self.row_index.len() * size_of::<StridesRowIndex>()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sparse<T> {
    mask: Arc<SparseMask>,
    data: Array1<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum SparseError {
    #[error("given matrix is too wide, must not exceed {}", Count::MAX)]
    TooWide,
    #[error(
        "given matrix has too many points, must not exceed {}",
        TotalCount::MAX
    )]
    TooManyPoints,
}

#[derive(thiserror::Error, Debug)]
pub enum SparseCheckError {
    #[error("given mask and data array do not agree on the total number of data points")]
    NonMatchingDataCount,
}

#[derive(Debug)]
pub struct SparseStats {
    pub mask_bytes_shareable: usize,
    pub mask_strong_ref_count: usize,
    pub data_bytes: usize,
}

impl<T> Sparse<T> {
    pub fn from_mask_and_data(
        mask: Arc<SparseMask>,
        data: Array1<T>,
    ) -> Result<Self, SparseCheckError> {
        let expected = mask.total_count_data() as usize;
        if expected != data.len() {
            return Err(SparseCheckError::NonMatchingDataCount);
        }
        Ok(Self { mask, data })
    }

    pub fn from_view(value: ArrayView2<T>, is_null: impl Fn(T) -> bool) -> Result<Self, SparseError>
    where
        T: Copy,
    {
        let (height, width) = value.dim();
        if width > Count::MAX as usize {
            return Err(SparseError::TooWide);
        }
        if width * height > u32::MAX as usize {
            return Err(SparseError::TooManyPoints);
        }
        let shape = [height, width];
        let mut data = Vec::new();
        let mut strides = Vec::new();
        let mut row_index = Vec::new();
        let mut strides_start_i = 0;
        for row_i in 0..height {
            let data_start_i = data.len();
            let row = value.row(row_i);
            let row = row
                .as_slice()
                .expect("row 0 is always contiguous? XX ah but order may not be standard");
            assert_eq!(row.len(), width);

            let mut stride = Stride::new();
            for col_i in 0..width {
                let val = row[col_i];
                if is_null(val) {
                    match stride.add_null() {
                        StrideAction::Noop => (),
                        StrideAction::Push(stride) => {
                            strides.push(stride);
                        }
                    }
                } else {
                    data.push(val);
                    // Can't overflow since we checked width <= Count::MAX
                    stride.count_data += 1;
                }
            }
            if !stride.is_empty() {
                strides.push(stride);
            }

            let strides_end_i = u32::try_from(strides.len())
                .expect("can't happen since we checked that width * height <= u32::MAX");
            row_index.push(StridesRowIndex {
                strides_start_i,
                strides_len: strides_end_i - strides_start_i,
                data_start_i,
            });
            strides_start_i = strides_end_i;
        }

        let mask = SparseMask {
            shape,
            strides: strides.into(),
            row_index: row_index.into(),
            total_count_data: data
                .len()
                .try_into()
                .expect("always succeeds because TotalCount is twice as wide as Count"),
        };
        Ok(Sparse {
            mask: mask.into(),
            data: data.into(),
        })
    }

    /// How many bytes of data this instance uses (malloc overheads
    /// not included). The mask can be shared between multiple
    /// instances, thus its `mask_bytes_shareable` value needs to be
    /// divided by `mask_strong_ref_count` to get the relevant cost.
    pub fn stats(&self) -> SparseStats {
        SparseStats {
            mask_bytes_shareable: self.mask.stats_bytes(),
            mask_strong_ref_count: Arc::strong_count(&self.mask),
            data_bytes: self.data.len() * size_of::<T>(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.mask.shape()
    }

    pub fn row_len(&self) -> usize {
        self.mask.row_len()
    }

    pub fn dim(&self) -> (usize, usize) {
        self.mask.dim()
    }

    pub fn num_values(&self) -> usize {
        self.mask.num_values()
    }

    pub fn width(&self) -> usize {
        self.mask.width()
    }

    pub fn height(&self) -> usize {
        self.mask.height()
    }

    /// Reconstructs row with index `row_i` by overwriting `row`.
    /// Panics for out of bounds `row_i` or invalid sizes of
    /// `row`. Returns the same reference as `row` but with now
    /// initialized type.
    #[inline]
    pub fn mut_slice_row_uninit(
        &self,
        row_i: usize,
        empty_val: T,
        row: &mut [MaybeUninit<T>],
    ) -> &mut [T]
    where
        T: Copy,
    {
        let row: &mut [T] = unsafe {
            // Safe because we're going to fill it completely, and we
            // are not reading from it.
            transmute(row)
        };
        self.mut_slice_row(row_i, empty_val, row)
    }

    /// Same as `mut_slice_row_uninit` but for already-initialized
    /// slices. Overwrites the slice completely.
    pub fn mut_slice_row<'a>(&self, row_i: usize, empty_val: T, row: &'a mut [T]) -> &'a mut [T]
    where
        T: Copy,
    {
        assert_eq!(row.len(), self.row_len());
        let StridesRowIndex {
            strides_start_i,
            strides_len,
            data_start_i,
        } = &self.mask.row_index[row_i];
        let strides = {
            let (i, len) = (*strides_start_i as usize, *strides_len as usize);
            &self.mask.strides[i..(i + len)]
        };
        let mut data_i = *data_start_i;
        let data = self.data.as_slice().expect("1D always succeeds");
        let mut col_i = 0;
        for Stride {
            count_empty,
            count_data,
        } in strides
        {
            {
                let count_empty = *count_empty as usize;
                if count_empty > 0 {
                    let col_i2 = col_i + count_empty;
                    row[col_i..col_i2].fill(empty_val);
                    col_i = col_i2;
                }
            }
            {
                let data_len = *count_data as usize;
                if data_len > 0 {
                    let col_i2 = col_i + data_len;
                    let row_sl = &mut row[col_i..col_i2];
                    let data_sl = &data[data_i..(data_i + data_len)];
                    row_sl.copy_from_slice(data_sl);
                    col_i = col_i2;
                    data_i += data_len;
                }
            }
        }
        row
    }

    /// Reconstructs row with index `row_i` into `row`, which should
    /// be empty when calling. Panics for out of bounds `row_i`.
    pub fn mut_row(&self, row_i: usize, empty_val: T, row: &mut Vec<T>)
    where
        T: Copy,
    {
        let StridesRowIndex {
            strides_start_i,
            strides_len,
            data_start_i,
        } = &self.mask.row_index[row_i];
        let strides = {
            let (i, len) = (*strides_start_i as usize, *strides_len as usize);
            &self.mask.strides[i..(i + len)]
        };
        let data = self.data.as_slice().expect("1D always succeeds");
        let mut data_i = *data_start_i;
        for Stride {
            count_empty,
            count_data,
        } in strides
        {
            {
                let count_empty = *count_empty as usize;
                if count_empty > 0 {
                    row.resize(row.len() + count_empty, empty_val);
                }
            }
            {
                let data_len = *count_data as usize;
                let sl = &data[data_i..(data_i + data_len)];
                row.extend_from_slice(sl);
                data_i += data_len;
            }
        }
    }

    pub fn row(&self, row_i: usize, empty_val: T) -> Box<[T]>
    where
        T: Copy,
    {
        let mut row = Vec::with_capacity(self.row_len());
        self.mut_row(row_i, empty_val, &mut row);
        row.into()
    }

    pub fn rows<'s>(&'s self, empty_val: T) -> impl DoubleEndedIterator<Item = Box<[T]>> + 's
    where
        T: Copy,
    {
        (0..self.mask.row_index.len()).map(move |row_i| self.row(row_i, empty_val))
    }

    pub fn decompress(&self, empty_val: T) -> Array2<T>
    where
        T: Copy,
    {
        let mut vs: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(self.num_values());
        for row_i in 0..self.height() {
            self.mut_slice_row_uninit(
                row_i,
                empty_val,
                &mut vs[row_i * self.width()..(row_i + 1) * self.width()],
            );
        }
        let vec: Vec<T> = unsafe {
            // Safe because we have completely overwritten every row.
            vs.assume_init()
        }
        .into();
        Array2::from_shape_vec(self.dim(), vec).expect("everything should match up")
    }

    /// Overwrite row with index `row_i` with the data in `row`, which
    /// is the uncompressed form of that row. The data from the pixels
    /// that are non-empty is copied. Emptiness is decided as per
    /// construction time, i.e. compression (or the 'mask' for empty
    /// points) does not change; points that were empty when
    /// constructing the mask are simply skipped here, regardless of
    /// value. Panics if `row.len()` does not match the width of the
    /// matrix.
    pub fn write_row_uncompressed(&mut self, row_i: usize, row: &[T])
    where
        T: Copy,
    {
        assert_eq!(row.len(), self.row_len());
        let StridesRowIndex {
            strides_start_i,
            strides_len,
            data_start_i,
        } = &self.mask.row_index[row_i];
        let strides = {
            let (i, len) = (*strides_start_i as usize, *strides_len as usize);
            &self.mask.strides[i..(i + len)]
        };
        let mut data_i = *data_start_i;
        let data = self.data.as_slice_mut().expect("1D always succeeds");
        let mut col_i = 0;
        for Stride {
            count_empty,
            count_data,
        } in strides
        {
            {
                let count_empty = *count_empty as usize;
                col_i += count_empty;
            }
            {
                let data_len = *count_data as usize;

                let col_i2 = col_i + data_len;
                let row_sl = &row[col_i..col_i2];
                let data_sl = &mut data[data_i..(data_i + data_len)];
                data_sl.copy_from_slice(row_sl);
                col_i = col_i2;
                data_i += data_len;
            }
        }
    }
}

use ndarray::{LinalgScalar, ScalarOperand};

macro_rules! def_array_binop {
    { $Op:tt, $method:tt, $op:tt } => {
        use std::ops::$Op;

        // With itself

        impl<T: LinalgScalar> $Op for &Sparse<T> {
            type Output = Sparse<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                let Sparse { mask, data } = self;
                let data = data $op &rhs.data;
                clone_arc!(mask);
                Sparse { mask, data }
            }
        }
        impl<T: LinalgScalar> $Op for Sparse<T> {
            type Output = Sparse<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                let Sparse { mask, data } = self;
                let data = data $op rhs.data;
                Sparse { mask, data }
            }
        }
        impl<T: LinalgScalar> $Op<&Sparse<T>> for Sparse<T> {
            type Output = Sparse<T>;

            fn $method(self, rhs: &Sparse<T>) -> Self::Output {
                let Sparse { mask, data } = self;
                let data = data $op &rhs.data;
                Sparse { mask, data }
            }
        }
        impl<T: LinalgScalar> $Op<Sparse<T>> for &Sparse<T> {
            type Output = Sparse<T>;

            fn $method(self, rhs: Sparse<T>) -> Self::Output {
                let Sparse { mask: _, data } = self;
                let data = data $op rhs.data;
                Sparse { mask: rhs.mask, data }
            }
        }

        // Any rhs for which $Op is implemented

        impl<T: LinalgScalar + ScalarOperand, T2: ScalarOperand> $Op<T2> for &Sparse<T>
        where
            for<'a> &'a Array1<T>: $Op<T2, Output = Array1<T>>,
        {
            type Output = Sparse<T>;

            fn $method(self, rhs: T2) -> Self::Output {
                let Sparse { mask, data } = self;
                let data = data $op rhs;
                clone_arc!(mask);
                Sparse { mask, data }
            }
        }
        impl<T: LinalgScalar + ScalarOperand, T2: ScalarOperand> $Op<T2> for Sparse<T>
        where
            Array1<T>: $Op<T2, Output = Array1<T>>,
        {
            type Output = Sparse<T>;

            fn $method(self, rhs: T2) -> Self::Output {
                let Sparse { mask, data } = self;
                let data = data $op rhs;
                Sparse { mask, data }
            }
        }
    }
}

def_array_binop! {Add, add, +}
def_array_binop! {Sub, sub, -}
def_array_binop! {Mul, mul, *}
def_array_binop! {Div, div, /}

#[cfg(test)]
mod tests {
    use anyhow::{anyhow, Result};
    use numpy::array;
    use rand::Rng;
    use rand_distr::Weibull;

    use crate::{dump::perhaps_dump_iteration_i, timing::show_current_timing};

    use super::*;

    fn test_correctness(a: Array2<i32>, height: usize) -> Result<Sparse<i32>> {
        dbg!(&a);
        let c = Sparse::from_view(a.view(), |x| x == 0)?;
        for (row_a, row_c) in a.rows().into_iter().zip(c.rows(0)) {
            let row_a = row_a.as_slice().expect("possible");
            assert_eq!(row_a, &*row_c);
        }
        assert_eq!(c.height(), height);
        assert_eq!(c.width(), a.row(0).dim());
        dbg!(&c);
        assert_eq!(&a, &c.decompress(0));
        Ok(c)
    }

    #[test]
    fn t_correctness() -> Result<()> {
        let a = array![
            [10, 11, 0, 14],
            [0, 0, 0, 13],
            [0, 12, 13, 14],
            [0, 0, 14, 0],
        ];
        test_correctness(a, 4)?;

        let a = array![[10, 11, 0, 14],];
        let mut c = test_correctness(a, 1)?;
        let mut r0 = c.row(0, 0);
        r0[1] += 1;
        r0[2] = 5;
        r0[3] = -5;
        assert_eq!(*r0, [10, 12, 5, -5]);
        assert_eq!(c.decompress(0), array![[10, 11, 0, 14],]);
        c.write_row_uncompressed(0, &*r0);
        assert_eq!(c.decompress(0), array![[10, 12, 0, -5],]);

        let a = array![[10], [0], [3],];
        test_correctness(a, 3)?;

        let a = array![[10, 11], [0, 0], [3, 0],];
        let mut c = test_correctness(a, 3)?;
        c.write_row_uncompressed(0, &[4, 14]);
        assert_eq!(c.decompress(0), array![[4, 14], [0, 0], [3, 0],]);
        c.write_row_uncompressed(2, &[16, 17]);
        c.write_row_uncompressed(1, &[15, 114]);
        c.write_row_uncompressed(0, &[5, 14]);
        assert_eq!(c.decompress(0), array![[5, 14], [0, 0], [16, 0],]);

        Ok(())
    }

    #[test]
    fn t_performance() -> Result<()> {
        let timing = show_current_timing(true, None, "rng".into());

        let dist = Weibull::new(1., 7.).unwrap();
        let mut rng = rand::thread_rng();
        let mut get_coord = |max_excl: usize| {
            let v = rng.sample(&dist);
            let xraw = (max_excl as f64 * v).abs() as isize - (max_excl as isize / 2);
            // dbg!((v, max, xraw));
            xraw.min(max_excl as isize - 1).abs() as usize
        };

        let width = 1000;
        let height = 800;

        let mut rng = rand::thread_rng();
        let mut ar = Array2::<f32>::zeros((height, width));
        for _ in 0..(2800 * width) {
            let a = get_coord(height - 1);
            let b = get_coord(width - 1);
            let lum = rng.gen_range((20.)..(25.));
            ar[(a, b)] = lum;
            ar[(a + 1, b)] = lum;
            ar[(a, b + 1)] = lum;
            ar[(a + 1, b + 1)] = lum;
        }
        // dbg!(&ar);
        perhaps_dump_iteration_i(0, 0, ar.view(), 0. ..25.6);

        show_current_timing(true, timing, "END".into());

        // Runs one decompression, compression and streaming
        // decompression
        let run_bench = move |c: &Sparse<f32>, run_no: u64, i: usize| -> Result<()> {
            let timing =
                show_current_timing(true, None, format!("{run_no}/{i}: decompress").into());

            let dec = c.decompress(0.);

            let timing =
                show_current_timing(true, timing, format!("{run_no}/{i}: compress").into());

            let c2 = Sparse::from_view(dec.view(), |x| x == 0.)?;

            let timing = show_current_timing(true, timing, format!("{run_no}/{i}: verify").into());

            assert_eq!(c, &c2);

            let timing = show_current_timing(
                true,
                timing,
                format!("{run_no}/{i}: streaming decompression").into(),
            );

            {
                let mut row = Vec::with_capacity(width);
                row.resize(width, 0.);
                for i in 0..height {
                    c.mut_slice_row(i, 0., &mut row);
                }
            }

            show_current_timing(true, timing, "end".into());

            dbg!((i, c.stats()));

            Ok(())
        };

        let c = Sparse::from_view(ar.view(), |x| x == 0.)?;
        {
            let dec = c.decompress(0.);
            assert_eq!(ar, dec);
        }

        let threads: Vec<_> = (1..32)
            .map(|i| {
                let mut c = c.clone();
                std::thread::spawn(move || -> Result<()> {
                    for j in 1..100 {
                        run_bench(&c, i, j)?;
                        c = c * 1.01234;
                    }
                    Ok(())
                })
            })
            .collect();

        for thread in threads {
            thread.join().map_err(|e| anyhow!("thread join: {e:?}"))??;
        }

        // panic!();

        Ok(())
    }
}
