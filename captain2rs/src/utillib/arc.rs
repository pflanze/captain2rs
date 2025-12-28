use std::sync::Arc;

pub trait CloneArc {
    fn clone_arc(&self) -> Self;
}

impl<T: ?Sized> CloneArc for Arc<T> {
    fn clone_arc(&self) -> Self {
        self.clone()
    }
}

#[macro_export]
macro_rules! clone_arc {
    { $id:ident } => {
        use $crate::utillib::arc::CloneArc;
        let $id = $id.clone_arc();
    }
}
