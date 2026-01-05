use std::{
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
};

pub trait EvalForCache<V, C>: Hash + Eq {
    fn eval(&self, context: &C) -> V;
}

#[derive(Debug, Default)]
pub struct EvaluationCache<K: EvalForCache<V, C>, V, C> {
    context: C,
    entries: HashMap<K, V>,
}

impl<K: EvalForCache<V, C>, V, C> EvaluationCache<K, V, C> {
    pub fn new(context: C) -> Self {
        Self {
            context,
            entries: HashMap::new(),
        }
    }

    pub fn get<'s>(&'s mut self, key: &K) -> &'s V
    where
        K: Clone,
    {
        if let Some(val) = self.entries.get(key) {
            let ptr: *const V = val;
            return unsafe {
                // Safe because the only reason for needing the ptr is
                // to work around the branch tracking limitation of
                // the current borrow checker (this branch would
                // compile fine without).
                &*ptr
            };
        }
        let val = key.eval(&self.context);
        self.entries.insert(key.clone(), val);
        &self.entries[key]
    }

    pub fn get_owned(&mut self, key: K) -> &V {
        match self.entries.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let val = occupied_entry.get();
                let ptr: *const V = val;
                unsafe {
                    // SHOULD be safe, right???  Why does std lib tie
                    // the lifetime of the result of get() to the
                    // Entry instead of the borrow of the HashMap???
                    &*ptr
                }
            }
            Entry::Vacant(vacant_entry) => {
                let val = vacant_entry.key().eval(&self.context);
                vacant_entry.insert(val)
            }
        }
    }

    /// Finish filling entries on the fly, offer as a fixed HashMap
    /// instead. (XX wrap to avoid mut?)
    pub fn closed(self) -> HashMap<K, V> {
        self.entries
    }
}
