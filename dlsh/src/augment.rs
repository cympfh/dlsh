use std::env;

mod data;
use data::Matrix;

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

impl Augment {
    fn forward(&self, x: &Matrix, opt: &Opts) -> Matrix {
        let (h, w) = x.shape();
        let noise = Matrix::random(h, w, opt.var);
        x + &noise
    }
}

fn main() {
    let opt = Opts::from_args();
    let nan_debug = env::var("NAN_DEBUG").is_ok();

    let x = Matrix::read();
    let (h, w) = x.shape();

    let strategy = match opt.strategy.as_ref() {
        "noise" | "gaussnoise" | "gauss_noise" | "GaussNoise" => Augment::GaussNoise,
        _ => panic!(format!("Unknown augment strategy: {}", opt.strategy)),
    };

    if nan_debug {
        let num: usize = x
            .data
            .iter()
            .map(|row| row.data.iter().filter(|val| val.is_nan()).count())
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

    let y = strategy.forward(&x, &opt);
    y.write();

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
        let gy = Matrix::read_from_file(&gradin_file);
        if let Some(gradout_file) = opt.gradout {
            gy.write_to_file(&gradout_file);
        }
    }
}
