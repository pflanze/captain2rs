//! 2D arrays compressed by omitting points that satisfy some
//! criterium (sparse matrices), optimized for the case of maps with
//! large run lengths, implemented by way of a simplified run-length
//! encoding (only the sparse points are encoded, other points are
//! repeated even if duplicates).

use std::{
    fmt::Debug,
    mem::{transmute, MaybeUninit},
    sync::Arc,
};

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Zero;

use crate::clone_arc;

type Count = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stride {
    count_empty: Count,
    count_data: Count,
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
    strides_start_i: u32,
    strides_len: u32,
    data_start_i: usize,
}

/// The information about where values in a compressed array belong
/// to.
#[derive(Debug)]
pub struct Compressed2Metadata {
    shape: [usize; 2],
    /// The sum of all counts
    strides: Box<[Stride]>,
    /// To get the slices of `strides` and `data`
    row_index: Box<[StridesRowIndex]>,
}

impl PartialEq for Compressed2Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.strides == other.strides
    }
}

impl Eq for Compressed2Metadata {}

impl Compressed2Metadata {
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
pub struct Compressed2<T> {
    metadata: Arc<Compressed2Metadata>,
    data: Array1<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum Compressed2Error {
    #[error("given matrix is too wide, must not exceed {}", Count::MAX)]
    TooWide,
    #[error("given matrix has too many points, must not exceed {}", u32::MAX)]
    TooManyPoints,
}

#[derive(Debug)]
pub struct CompressedStats {
    pub metadata_bytes_shareable: usize,
    pub metadata_strong_ref_count: usize,
    pub data_bytes: usize,
}

impl<T> Compressed2<T> {
    pub fn from_view(
        value: ArrayView2<T>,
        is_null: impl Fn(T) -> bool,
    ) -> Result<Self, Compressed2Error>
    where
        T: Copy,
    {
        let (height, width) = value.dim();
        if width > Count::MAX as usize {
            return Err(Compressed2Error::TooWide);
        }
        if width * height > u32::MAX as usize {
            return Err(Compressed2Error::TooManyPoints);
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

        let metadata = Compressed2Metadata {
            shape,
            strides: strides.into(),
            row_index: row_index.into(),
        };
        Ok(Compressed2 {
            metadata: metadata.into(),
            data: data.into(),
        })
    }

    /// How many bytes of data this instance uses (malloc overheads
    /// not included). Metadata can be shared between multiple
    /// instances, thus its `metadata_bytes_shareable` value needs to
    /// be divided by `metadata_strong_ref_count` to get the relevant
    /// cost.
    pub fn stats(&self) -> CompressedStats {
        CompressedStats {
            metadata_bytes_shareable: self.metadata.stats_bytes(),
            metadata_strong_ref_count: Arc::strong_count(&self.metadata),
            data_bytes: self.data.len() * size_of::<T>(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.metadata.shape()
    }

    pub fn row_len(&self) -> usize {
        self.metadata.row_len()
    }

    pub fn dim(&self) -> (usize, usize) {
        self.metadata.dim()
    }

    pub fn num_values(&self) -> usize {
        self.metadata.num_values()
    }

    pub fn width(&self) -> usize {
        self.metadata.width()
    }

    pub fn height(&self) -> usize {
        self.metadata.height()
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
        } = &self.metadata.row_index[row_i];
        let strides = {
            let (i, len) = (*strides_start_i as usize, *strides_len as usize);
            &self.metadata.strides[i..(i + len)]
        };
        let mut data_i = *data_start_i;
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
                let data = self.data.as_slice().expect("1D always succeeds");
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
        } = &self.metadata.row_index[row_i];
        let strides = {
            let (i, len) = (*strides_start_i as usize, *strides_len as usize);
            &self.metadata.strides[i..(i + len)]
        };
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
                let data = self.data.as_slice().expect("1D always succeeds");
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
        (0..self.metadata.row_index.len()).map(move |row_i| self.row(row_i, empty_val))
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
}

use ndarray::{LinalgScalar, ScalarOperand};

macro_rules! def_array_binop {
    { $Op:tt, $method:tt, $op:tt } => {
        use std::ops::$Op;

        // With itself

        impl<T: LinalgScalar> $Op for &Compressed2<T> {
            type Output = Compressed2<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                let Compressed2 { metadata, data } = self;
                let data = data $op &rhs.data;
                clone_arc!(metadata);
                Compressed2 { metadata, data }
            }
        }
        impl<T: LinalgScalar> $Op for Compressed2<T> {
            type Output = Compressed2<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                let Compressed2 { metadata, data } = self;
                let data = data $op rhs.data;
                Compressed2 { metadata, data }
            }
        }
        impl<T: LinalgScalar> $Op<&Compressed2<T>> for Compressed2<T> {
            type Output = Compressed2<T>;

            fn $method(self, rhs: &Compressed2<T>) -> Self::Output {
                let Compressed2 { metadata, data } = self;
                let data = data $op &rhs.data;
                Compressed2 { metadata, data }
            }
        }
        impl<T: LinalgScalar> $Op<Compressed2<T>> for &Compressed2<T> {
            type Output = Compressed2<T>;

            fn $method(self, rhs: Compressed2<T>) -> Self::Output {
                let Compressed2 { metadata: _, data } = self;
                let data = data $op rhs.data;
                Compressed2 { metadata: rhs.metadata, data }
            }
        }

        // Any rhs for which $Op is implemented

        impl<T: LinalgScalar + ScalarOperand, T2: ScalarOperand> $Op<T2> for &Compressed2<T>
        where
            for<'a> &'a Array1<T>: $Op<T2, Output = Array1<T>>,
        {
            type Output = Compressed2<T>;

            fn $method(self, rhs: T2) -> Self::Output {
                let Compressed2 { metadata, data } = self;
                let data = data $op rhs;
                clone_arc!(metadata);
                Compressed2 { metadata, data }
            }
        }
        impl<T: LinalgScalar + ScalarOperand, T2: ScalarOperand> $Op<T2> for Compressed2<T>
        where
            Array1<T>: $Op<T2, Output = Array1<T>>,
        {
            type Output = Compressed2<T>;

            fn $method(self, rhs: T2) -> Self::Output {
                let Compressed2 { metadata, data } = self;
                let data = data $op rhs;
                Compressed2 { metadata, data }
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

    use crate::{dump::perhaps_dump, timing::show_current_timing};

    use super::*;

    fn test_correctness(a: Array2<i32>, height: usize) -> Result<()> {
        dbg!(&a);
        let c = Compressed2::from_view(a.view(), |x| x == 0)?;
        for (row_a, row_c) in a.rows().into_iter().zip(c.rows(0)) {
            let row_a = row_a.as_slice().expect("possible");
            assert_eq!(row_a, &*row_c);
        }
        assert_eq!(c.height(), height);
        assert_eq!(c.width(), a.row(0).dim());
        dbg!(&c);
        assert_eq!(&a, &c.decompress(0));
        Ok(())
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
        test_correctness(a, 1)?;

        let a = array![[10], [0], [3],];
        test_correctness(a, 3)?;

        let a = array![[10, 11], [0, 0], [3, 0],];
        test_correctness(a, 3)?;

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
        perhaps_dump(0, 0, ar.view());

        show_current_timing(true, timing, "END".into());

        // Runs one decompression, compression and streaming
        // decompression
        let run_bench = move |c: &Compressed2<f32>, run_no: u64, i: usize| -> Result<()> {
            let timing =
                show_current_timing(true, None, format!("{run_no}/{i}: decompress").into());

            let dec = c.decompress(0.);

            let timing =
                show_current_timing(true, timing, format!("{run_no}/{i}: compress").into());

            let c2 = Compressed2::from_view(dec.view(), |x| x == 0.)?;

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

        let c = Compressed2::from_view(ar.view(), |x| x == 0.)?;
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
