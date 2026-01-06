pub trait IdAsIndex: From<usize> + Send + Sync + 'static {
    fn id_as_index(&self) -> usize;
}

#[macro_export]
macro_rules! def_id {
    { { $($pub:tt)* } { $name:ident } } => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        $($pub)* struct $name(usize);

        impl $crate::id::IdAsIndex for $name {
            fn id_as_index(&self) -> usize {
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
        def_id!{{pub} {$name}}
    };
    { pub($pub_args:tt) $name:ident } => {
        def_id!{{pub($pub_args)} {$name}}
    };
    { $name:ident } => {
        def_id!{{} {$name}}
    }
}
