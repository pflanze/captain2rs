use anyhow::{bail, Result};

pub struct Coo4 {
    points: Vec<((u16, u16, u16, u16), f64)>,
    is_sorted: bool,
}

impl Coo4 {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            is_sorted: true,
        }
    }

    fn _insert<const REPORT_ERROR: bool>(
        &mut self,
        coords: (u16, u16, u16, u16),
        val: f64,
    ) -> Result<()> {
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
    pub fn insert(&mut self, coords: (u16, u16, u16, u16), val: f64) -> Result<()> {
        self._insert::<true>(coords, val)
    }

    /// Does not gives an error if the coordinates are not after those
    /// lsat inserted, but marks the tensor as unsorted (must be
    /// sorted before it can be used for retrievals)
    pub fn insert_unchecked(&mut self, coords: (u16, u16, u16, u16), val: f64) {
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
    pub fn get(&self, coords: (u16, u16, u16, u16)) -> Option<f64> {
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
