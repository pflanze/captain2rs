use anyhow::{anyhow, Result};
use captain2rs::{
    biodivsim::{div::Float, sparse_dispersal::SparseDispersal},
    clone, clone_arc, perhaps_dump,
    sparse::Sparse,
    timing::show_current_timing,
    utillib::arc::{CloneArc, IntoArc},
};
use ndarray::Array2;
use rand::Rng;
use rand_distr::Weibull;

fn main() -> Result<()> {
    // Parameters:
    let width = 1000;
    let height = 800;
    let lambda0 = 0.5;
    let threshold = 3;
    let equalize = false;
    let num_threads = 16; // I have 16 cores, 32 virtual threads

    // Partial copy-paste from sparse.rs

    let timing = show_current_timing(true, None, "rng generate pic".into());

    let ar = {
        let dist = Weibull::new(1., 7.).unwrap();
        let mut rng = rand::thread_rng();
        let mut get_coord = |max_excl: usize| {
            let v = rng.sample(&dist);
            let xraw = (max_excl as f32 * v).abs() as isize - (max_excl as isize / 2);
            // dbg!((v, max, xraw));
            xraw.min(max_excl as isize - 1).abs() as usize
        };

        let mut rng = rand::thread_rng();
        let mut ar = Array2::<Float>::zeros((height, width));
        for _ in 0..(2800 * width) {
            let a = get_coord(height - 1);
            let b = get_coord(width - 1);
            let lum = rng.gen_range((20.)..(25.));
            ar[(a, b)] = lum;
            ar[(a + 1, b)] = lum;
            ar[(a, b + 1)] = lum;
            ar[(a + 1, b + 1)] = lum;
        }
        ar
    };

    show_current_timing(true, timing, "END".into());

    // dbg!(&ar);
    perhaps_dump!("sparse-dispersal-benchmark_ar", ar.view(), 0. ..25.6);

    let timing = show_current_timing(true, None, "compress+dispersal".into());

    let c0 = Sparse::from_view_and_pred(ar.view(), |x| x == 0.)?.into_arc();
    let mask = c0.mask().clone_arc();
    let dispersal = SparseDispersal::new(lambda0, threshold, mask.clone_arc()).into_arc();

    show_current_timing(true, timing, "END".into());

    let run_bench = move |mut c: Sparse<Float>, run_no: u64, i: usize| -> Result<()> {
        let timing =
            show_current_timing(true, None, format!("apply 10 times in {run_no}/{i}").into());

        for _ in 0..10 {
            dispersal.apply_mut(&mut c, equalize)?;
        }

        show_current_timing(true, timing, "END".into());
        perhaps_dump!(
            format!("run-bench_c-{run_no}-{i}"),
            c.decompress(0.).view(),
            0. ..25.
        );
        Ok(())
    };

    run_bench((*c0).clone(), 0, 0)?;

    let threads: Vec<_> = (1..=num_threads)
        .map(|i| {
            clone_arc!(c0);
            clone!(run_bench);
            std::thread::spawn(move || -> Result<()> {
                for j in 1..=10 {
                    run_bench((*c0).clone(), i, j)?;
                }
                Ok(())
            })
        })
        .collect();

    for thread in threads {
        thread.join().map_err(|e| anyhow!("thread join: {e:?}"))??;
    }

    Ok(())
}
