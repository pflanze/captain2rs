//! Typed IDs and vectors using them as keys.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub trait IdVecIndex: From<usize> + Send + Sync + 'static {
    fn id_vec_index(&self) -> usize;
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IdVec<Idx: IdVecIndex, T> {
    _indexing_type: PhantomData<Idx>,
    vec: Vec<T>,
}

impl<Id: IdVecIndex, T> IdVec<Id, T> {
    pub fn par_iter_mut_enumerated(&mut self) -> impl IndexedParallelIterator<Item = (Id, &mut T)>
    where
        T: Send + Sync + 'static,
    {
        self.vec
            .as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .map(|(i, t)| (Id::from(i), t))
    }
}

// XX really allow this?
impl<Id: IdVecIndex, T> Deref for IdVec<Id, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

// XX really allow this?
impl<Id: IdVecIndex, T> DerefMut for IdVec<Id, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<Idx: IdVecIndex, T> Index<Idx> for IdVec<Idx, T> {
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vec[index.id_vec_index()]
    }
}

impl<Idx: IdVecIndex, T> IndexMut<Idx> for IdVec<Idx, T> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.vec[index.id_vec_index()]
    }
}

impl<Idx: IdVecIndex, T> From<Vec<T>> for IdVec<Idx, T> {
    fn from(vec: Vec<T>) -> Self {
        Self {
            _indexing_type: PhantomData,
            vec,
        }
    }
}

#[macro_export]
macro_rules! def_id_vec_id {
    { { $($pub:tt)* } { $name:ident } } => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        $($pub)* struct $name(usize);

        impl $crate::id_vec::IdVecIndex for $name {
            fn id_vec_index(&self) -> usize {
                self.0
            }
        }

        impl From<usize> for $name {
            fn from(value: usize) -> Self {
                $name(value)
            }
        }
    };
    { pub $name:ident } => {
        def_id_vec_id!{{pub} {$name}}
    };
    { pub($pub_args:tt) $name:ident } => {
        def_id_vec_id!{{pub($pub_args)} {$name}}
    };
    { $name:ident } => {
        def_id_vec_id!{{} {$name}}
    }
}
