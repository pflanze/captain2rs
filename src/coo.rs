use std::fmt::Debug;

use anyhow::{bail, Result};
use num_traits::PrimInt;

/// C is the type of the coordinate components, D is the
/// dimensionality, V is the value type.
pub struct Coo<C: PrimInt, const D: usize, V> {
    points: Vec<([C; D], V)>,
    is_sorted: bool,
}

impl<C: PrimInt + Debug, const D: usize, V> Coo<C, D, V> {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            is_sorted: true,
        }
    }

    fn _insert<const REPORT_ERROR: bool>(&mut self, coords: [C; D], val: V) -> Result<()> {
        if self.is_sorted || REPORT_ERROR {
            if let Some((last_coords, _)) = self.points.last() {
                if !(last_coords < &coords) {
                    if REPORT_ERROR {
                        bail!(
                            "not strictly increasing coordinates: {last_coords:?} vs. {coords:?}"
                        );
                    }
                    self.is_sorted = false;
                }
            }
        }
        self.points.push((coords, val));
        Ok(())
    }

    /// Gives an error if the coordinates are not after those lsat
    /// inserted.
    pub fn insert<C0>(&mut self, coords: [C0; D], val: V) -> Result<()>
    where
        C: From<C0>,
    {
        let coords = coords.map(From::from);
        self._insert::<true>(coords, val)
    }

    /// Does not gives an error if the coordinates are not after those
    /// lsat inserted, but marks the tensor as unsorted (must be
    /// sorted before it can be used for retrievals)
    pub fn insert_unchecked<C0>(&mut self, coords: [C0; D], val: V)
    where
        C: From<C0>,
    {
        let coords = coords.map(From::from);
        _ = self._insert::<false>(coords, val);
    }

    /// Check that the ordering is correct: a section must be
    /// sorted. The next lower level (next dimension) section per same
    /// value must be sorted again
    pub fn check(&self) -> Result<()> {
        if self.points.is_empty() {
            return Ok(());
        }

        let mut last_c = &self.points[0].0;
        for (c, _) in &self.points[1..] {
            if !(last_c < c) {
                if last_c == c {
                    bail!("coordinates set twice: {c:?}");
                }
                bail!("not strictly increasing coordinates: {last_c:?} vs. {c:?}");
            }
            last_c = c;
        }

        Ok(())
    }

    pub fn sort(&mut self) -> Result<()> {
        if !self.is_sorted {
            self.points.sort_by(|(c1, _), (c2, _)| c1.cmp(c2));
            // XX should  just 'do overwrites'  i.e. drop older entries, gll. b how  algo for sort.
            self.check()?;
            self.is_sorted = true;
        }
        Ok(())
    }

    pub fn is_sorted(&self) -> bool {
        self.is_sorted
    }

    /// Must only be called on a sorted self, otherwise panics!
    pub fn get_ref(&self, coords: [C; D]) -> Option<&V> {
        assert!(self.is_sorted);
        match self
            .points
            .binary_search_by_key(&coords, |(coords, _)| *coords)
        {
            Ok(i) => Some(&self.points[i].1),
            Err(_) => None,
        }
    }

    /// Must only be called on a sorted self, otherwise panics!
    pub fn get(&self, coords: [C; D]) -> Option<V>
    where
        V: Copy,
    {
        assert!(self.is_sorted);
        match self
            .points
            .binary_search_by_key(&coords, |(coords, _)| *coords)
        {
            Ok(i) => Some(self.points[i].1),
            Err(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}
