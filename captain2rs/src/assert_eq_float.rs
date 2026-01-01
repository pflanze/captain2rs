use std::fmt::Debug;

use num_traits::Float;

pub fn float_is_close<T: Float + Debug>(a: T, b: T) -> bool {
    if (a * b).is_sign_negative() {
        return false;
    }
    let d = if a > b { a / b } else { b / a };
    // XX this won't work for bf16 !
    let large_epsilon = T::from(100.).unwrap() * Float::epsilon();
    if d < T::one() + large_epsilon {
        return true;
    }
    // (a - b).abs() < large_epsilon
    false
}

#[test]
fn t_float_is_close() {
    let c = float_is_close::<f64>;
    assert!(c(191.439, 191.439));
    //      1.4142135623731
    assert!(1.0000000000000003 > 1.);
    assert!(c(1.0000000000000003, 1.));
    assert!(c(1., 1.0000000000000003));
    assert!(c(1. / 1.0000000000000003, 1.0000000000000003));
    assert!(!c(1. / 1.00000000000002, 1.00000000000002));
    assert!(c(1., 1.00000000000002));
    assert!(!c(1., -1.00000000000002));

    assert!(c(1.0e32, 1.00000000000002e32));
    assert!(!c(1.0e-32, 1.00000000000002e32));
    assert!(c(1.0e-32, 1.00000000000002e-32));
    assert!(!c(1.0e-32, 1.0000000000002e-32));
    assert!(!(1.0e-32 == 1.00000000000002e-32));
}

#[macro_export]
macro_rules! assert_eq_float {
    { $a:expr, $b:expr } => {
        let a = $a;
        let b = $b;
        if !$crate::assert_eq_float::float_is_close(a, b) {
            panic!("assert_eq_float!({}, {}): {a} is not close to {b}",
                   stringify!($a), stringify!($b))
        }
    }
}
