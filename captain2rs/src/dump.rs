use std::{
    env,
    fs::{create_dir, File},
    io::BufWriter,
    ops::Range,
};

use lazy_static::lazy_static;
use ndarray::ArrayView2;
use num_traits::{AsPrimitive, Float};

lazy_static! {
    static ref DUMP: bool = env::var_os("DUMP").is_some();
}

pub fn perhaps_dump<T: Float + AsPrimitive<u8>>(
    iteration: u64,
    i: usize,
    h: ArrayView2<T>,
    range: Range<T>,
) {
    if *DUMP {
        let path = format!("dump/{iteration:03}-{i:04}.png");
        let _ = create_dir("dump");
        let w = BufWriter::new(File::create(&path).unwrap());
        let (width, height) = h.dim();
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
                let val = h[(x, y)];
                data[i] = ((val - start) * factor).as_()
            }
        }
        writer.write_image_data(&data).unwrap(); // Save
    }
}
