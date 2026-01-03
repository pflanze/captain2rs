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
        let $id = {
            use $crate::utillib::arc::CloneArc;
            $id.clone_arc()
        };
    }
}

#[macro_export]
macro_rules! clone {
    { $id:ident } => {
        let $id = $id.clone();
    }
}

pub trait IntoArc {
    fn into_arc(self) -> Arc<Self>;
}

impl<T> IntoArc for T {
    fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}
