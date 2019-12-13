mod data;
use data::matrix;
use data::matrix::Matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    /// loss type (e.g. mse, kl)
    losstype: String,
    #[structopt(short = "t", long = "true", required = true)]
    truefile: String,
    #[structopt(short = "o", long = "out")]
    gradout: Option<String>,
}

#[derive(PartialEq, Eq)]
enum Loss {
    MSE,
    KL,
}

use Loss::*;

impl Loss {
    fn forward(&self, x: f32, y: f32) -> f32 {
        match self {
            MSE => {
                (x - y).powi(2)
            },
            KL => {
                x * (x.ln() - y.ln())
            },
        }
    }
    fn backward(&self, x: f32, y: f32) -> f32 {
        match self {
            MSE => {
                x - y
            },
            KL => {
                x.ln() - y.ln() + 1.0
            },
        }
    }

}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();
    let (h, w) = matrix::shape(&x);

    let mut y = matrix::read_from_file(&opt.truefile);
    if matrix::shape(&y) != (h, w) {
        panic!(format!(
            "Imcompatible shape. Can accept same shape matrices: shape(input) = {:?}, shape(true) = {:?}",
            matrix::shape(&x),
            matrix::shape(&y)));
    }

    let loss = match opt.losstype.as_ref() {
        "mse" => MSE,
        "kl" => KL,
        _ => {
            panic!(format!("Unknown loss type: {}", opt.losstype))
        },
    };

    if loss == KL {
        // MUST: y > eps; sum y = 1.0
        let eps: f32 = 0.0000001;
        for i in 0..h {
            let mut sum = 0.0;
            for j in 0..w {
                if y[i][j] < eps {
                    y[i][j] = eps;
                }
                sum += y[i][j];
            }
            for j in 0..w {
                y[i][j] /= sum;
            }
        }
    }

    let z: Matrix = (0..h).map(|i|
        (0..w).map(|j|
            loss.forward(x[i][j], y[i][j])
        ).collect()
    ).collect();

    matrix::write(&z);

    if let Some(gradout_file) = opt.gradout {
        let g: Matrix = (0..h).map(|i|
            (0..w).map(|j|
                loss.backward(x[i][j], y[i][j])
            ).collect()
        ).collect();
        matrix::write_to_file(&g, &gradout_file);
    }

}
