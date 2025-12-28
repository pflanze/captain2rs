use std::sync::Arc;

pub trait CloneArc {
    fn clone_arc(&self) -> Self;
}

impl<T: ?Sized> CloneArc for Arc<T> {
    fn clone_arc(&self) -> Self {
        self.clone()
    }
}
