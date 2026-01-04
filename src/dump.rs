use std::{
    env,
    fs::{File, create_dir},
    io::BufWriter,
    ops::Range,
    path::PathBuf,
};

use lazy_static::lazy_static;
use ndarray::ArrayView2;
use num_traits::{AsPrimitive, Float};

lazy_static! {
    pub static ref DUMP: bool = env::var_os("DUMP").is_some();
}

pub fn _perhaps_dump<T: Float + AsPrimitive<u8>>(
    mut file_name_without_suffix: String,
    h: ArrayView2<T>,
    range: Range<T>,
) {
    if *DUMP {
        let path = {
            let mut path = PathBuf::from("dump");
            let _ = create_dir(&path);
            file_name_without_suffix.push_str(".png");
            path.push(&file_name_without_suffix);
            path
        };
        let w = BufWriter::new(File::create(&path).unwrap());
        let (height, width) = h.dim();
        let mut encoder =
            png::Encoder::new(w, width.try_into().unwrap(), height.try_into().unwrap());
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let mut data: Vec<u8> = vec![0; width * height];
        let Range { start, end } = range;
        let factor = T::from(256).unwrap() / (end - start);
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let val = h[(y, x)];
                data[i] = ((val - start) * factor).as_()
            }
        }
        writer.write_image_data(&data).unwrap(); // Save
    }
}

pub fn perhaps_dump_iteration_i<T: Float + AsPrimitive<u8>>(
    iteration: u64,
    i: usize,
    h: ArrayView2<T>,
    range: Range<T>,
) {
    if *DUMP {
        _perhaps_dump(format!("{iteration:03}-{i:04}"), h, range);
    }
}

pub fn perhaps_dump_name_i<T: Float + AsPrimitive<u8>>(
    name: &str,
    i: usize,
    h: ArrayView2<T>,
    range: Range<T>,
) {
    if *DUMP {
        _perhaps_dump(format!("{name}-{i:04}"), h, range);
    }
}

/// `gen_name` must return a file name without suffix nor slashes.
pub fn perhaps_dump_<T: Float + AsPrimitive<u8>, S: Into<String>>(
    gen_name: impl FnOnce() -> S,
    h: ArrayView2<T>,
    range: Range<T>,
) {
    if *DUMP {
        _perhaps_dump(gen_name().into(), h, range);
    }
}

#[macro_export]
macro_rules! perhaps_dump {
    { $name:expr, $h:expr, $range:expr } => {
        if *$crate::dump::DUMP {
            $crate::dump::_perhaps_dump($name.into(), $h, $range);
        }
    }
}
