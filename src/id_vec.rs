//! Typed IDs and vectors using them as keys.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::id::IdAsIndex;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IdVec<Idx: IdAsIndex, T> {
    _indexing_type: PhantomData<Idx>,
    vec: Vec<T>,
}

impl<Id: IdAsIndex, T> IdVec<Id, T> {
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
impl<Id: IdAsIndex, T> Deref for IdVec<Id, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

// XX really allow this?
impl<Id: IdAsIndex, T> DerefMut for IdVec<Id, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<Idx: IdAsIndex, T> Index<Idx> for IdVec<Idx, T> {
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vec[index.id_as_index()]
    }
}

impl<Idx: IdAsIndex, T> IndexMut<Idx> for IdVec<Idx, T> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.vec[index.id_as_index()]
    }
}

impl<Idx: IdAsIndex, T> From<Vec<T>> for IdVec<Idx, T> {
    fn from(vec: Vec<T>) -> Self {
        Self {
            _indexing_type: PhantomData,
            vec,
        }
    }
}
