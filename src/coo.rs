use std::{fmt::Debug, ops::Index};

use anyhow::{Result, bail};
use num_traits::PrimInt;

/// C is the type of the coordinate components, D is the
/// dimensionality, V is the value type.
#[derive(Debug, Clone)]
pub struct Coo<C: PrimInt, const D: usize, V> {
    default: V,
    points: Vec<([C; D], V)>,
    is_sorted: bool,
}

impl<C: PrimInt + Debug, const D: usize, V> Coo<C, D, V> {
    pub fn new(default: V) -> Self {
        Self {
            default,
            points: Vec::new(),
            is_sorted: true,
        }
    }

    pub fn fill_value(&self) -> &V {
        &self.default
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
    /// last inserted, but marks the tensor as unsorted (it must then
    /// be sorted before it can be used for retrievals or
    /// overwrites). Note: does *not* allow overwrites: writing to a
    /// coordinate that was already set will lead to a duplicate value
    /// error the next time the tensor is sorted.
    pub fn insert_unordered<C0>(&mut self, coords: [C0; D], val: V)
    where
        C: From<C0>,
    {
        let coords = coords.map(From::from);
        _ = self._insert::<false>(coords, val);
    }

    /// Overwrites an existing value at the given coordinates and
    /// returns true if so. If there is no value, does nothing and
    /// returns false. Note: forces the tensor to be sorted; thus, if
    /// efficiency matters, don't interleave this call with
    /// `insert_unordered` or `set`.
    pub fn overwrite<C0>(&mut self, coords: [C0; D], val: V) -> Result<bool>
    where
        C: From<C0>,
    {
        self.sort()?;
        if let Some(rf) = self.get_mut(coords) {
            *rf = val;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Insert a value, overwriting a previously inserted one if
    /// present. Returns `false` if there was no slot for the given
    /// coordinates, `true` if one was overwritten. Note: forces the
    /// tensor to be sorted, but then leaves it unsorted if the value
    /// was newly inserted; thus, if efficiency matters, don't
    /// interleave this call with `insert_unordered`, and avoid
    /// calling it again if it returned `false`.
    pub fn set<C0: Copy>(&mut self, coords: [C0; D], val: V) -> Result<bool>
    where
        C: From<C0>,
        V: Clone,
    {
        if self.overwrite(coords, val.clone())? {
            return Ok(true);
        }
        self.insert_unordered(coords, val);
        Ok(false)
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
            self.points.sort_unstable_by(|(c1, _), (c2, _)| c1.cmp(c2));
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
    pub fn get_mut<C0>(&mut self, coords: [C0; D]) -> Option<&mut V>
    where
        C: From<C0>,
    {
        assert!(self.is_sorted);
        let coords = coords.map(|c| c.into());
        match self
            .points
            .binary_search_by_key(&coords, |(coords, _)| *coords)
        {
            Ok(i) => Some(&mut self.points[i].1),
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

    pub fn to_coords_and_values(&self) -> (Vec<C>, Vec<V>)
    where
        C: Copy,
        V: Clone,
    {
        let mut coords: Vec<C> = Vec::new();
        let mut vals = Vec::new();
        for (coord, val) in &self.points {
            coords.push(coord[0]);
            vals.push(val.clone());
        }
        for i in 1..D {
            for (coord, _) in &self.points {
                coords.push(coord[i]);
            }
        }
        (coords, vals)
    }
}

impl<C: PrimInt + Debug, const D: usize, V> Index<[C; D]> for Coo<C, D, V> {
    type Output = V;

    fn index(&self, index: [C; D]) -> &Self::Output {
        self.get_ref(index).unwrap_or(&self.default)
    }
}
