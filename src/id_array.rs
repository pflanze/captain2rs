//! ndarray arrays in which the first dimension is of a custom (id)
//! type (the other dimensions are of type usize).

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use ndarray::{ArrayView0, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::id::IdAsIndex;

macro_rules! def_id_array_1plus {
    { $IdArray_:tt : $Array_:tt } =>  {

        use ndarray::$Array_;

        #[derive(Debug, Default, Clone, PartialEq, Eq)]
        pub struct $IdArray_<Idx: IdAsIndex, T> {
            _indexing_type: PhantomData<Idx>,
            array: $Array_<T>,
        }

        impl<Id: IdAsIndex, T> $IdArray_<Id, T> {
            /// Returns None if the underlying array cannot provide a slice
            pub fn par_iter_mut_enumerated(
                &mut self,
            ) -> Option<impl IndexedParallelIterator<Item = (Id, &mut T)>>
            where
                T: Send + Sync + 'static,
            {
                Some(
                    self.array
                        .as_slice_mut()?
                        .par_iter_mut()
                        .enumerate()
                        .map(|(i, t)| (Id::from(i), t)),
                )
            }
        }

        // XX really allow this?
        impl<Id: IdAsIndex, T> Deref for $IdArray_<Id, T> {
            type Target = $Array_<T>;

            fn deref(&self) -> &Self::Target {
                &self.array
            }
        }

        // XX really allow this?
        impl<Id: IdAsIndex, T> DerefMut for $IdArray_<Id, T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.array
            }
        }

        impl<Idx: IdAsIndex, T> From<$Array_<T>> for $IdArray_<Idx, T> {
            fn from(array: $Array_<T>) -> Self {
                Self {
                    _indexing_type: PhantomData,
                    array,
                }
            }
        }
    }
}

macro_rules! def_id_array_2plus {
    { $IdArray_:tt : $Array_:tt, at: $At:ty } => {
        def_id_array_1plus!{ $IdArray_: $Array_ }

        impl<Id: IdAsIndex, T> $IdArray_<Id, T> {
            pub fn at(&self, id: Id) -> $At {
                self.array.index_axis(Axis(0), id.id_as_index())
            }
        }
    }
}

def_id_array_2plus! { IdArray1: Array1, at: ArrayView0<'_, T> }

impl<Idx: IdAsIndex, T> Index<Idx> for IdArray1<Idx, T> {
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.array[index.id_as_index()]
    }
}

impl<Idx: IdAsIndex, T> IndexMut<Idx> for IdArray1<Idx, T> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.array[index.id_as_index()]
    }
}

impl<Idx: IdAsIndex, T> From<Vec<T>> for IdArray1<Idx, T> {
    fn from(vec: Vec<T>) -> Self {
        Self {
            _indexing_type: PhantomData,
            array: vec.into(),
        }
    }
}

def_id_array_2plus! { IdArray2: Array2, at: ArrayView1<'_, T> }
def_id_array_2plus! { IdArray3: Array3, at: ArrayView2<'_, T> }
def_id_array_2plus! { IdArray4: Array4, at: ArrayView3<'_, T> }
