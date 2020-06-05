use rand_distr::{Distribution, Normal};
use std::env;

mod data;
use data::matrix;
use data::matrix::Matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    /// Augment Strategy
    strategy: String,
    #[structopt(short = "v", long = "var", default_value = "0.1")]
    var: f32,
    #[structopt(short = "i", long = "in")]
    gradin: Option<String>,
    #[structopt(short = "o", long = "out")]
    gradout: Option<String>,
}

#[derive(Debug)]
enum Augment {
    GaussNoise,
}

fn main() {
    let opt = Opts::from_args();
    let nan_debug = env::var("NAN_DEBUG").is_ok();

    let x = matrix::read();
    let (h, w) = matrix::shape(&x);

    let strategy = match opt.strategy.as_ref() {
        "noise" | "gaussnoise" | "gauss_noise" | "GaussNoise" => Augment::GaussNoise,
        _ => panic!(format!("Unknown augment strategy: {}", opt.strategy)),
    };

    if nan_debug {
        let num: usize = x
            .iter()
            .map(|row| row.iter().filter(|val| val.is_nan()).count())
            .sum();
        if num > 0 {
            eprintln!(
                "[{:?}] NaN detected from Input ({} / {:?})",
                strategy,
                num,
                (h, w)
            );
            panic!();
        }
    }

    let normal = Normal::new(0.0, opt.var).unwrap();
    let y: Matrix = (0..h)
        .map(|i| {
            (0..w)
                .map(|j| {
                    let noise = normal.sample(&mut rand::thread_rng());
                    x[i][j] + noise
                })
                .collect()
        })
        .collect();
    matrix::write(&y);

    if nan_debug {
        for i in 0..h {
            for j in 0..w {
                if y[i][j].is_nan() {
                    eprintln!(
                        "[{:?}] NaN detected from Output; forward({}) = NaN",
                        strategy, x[i][j],
                    );
                    eprintln!("[{:?}] x[] = {:?}", strategy, &x[i]);
                    panic!();
                }
            }
        }
    }

    if let Some(gradin_file) = opt.gradin {
        let gy = matrix::read_from_file(&gradin_file);
        if let Some(gradout_file) = opt.gradout {
            matrix::write_to_file(&gy, &gradout_file);
        }
    }
}
