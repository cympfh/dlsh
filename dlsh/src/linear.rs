mod data;
use data::matrix;
use data::matrix::Matrix;

use std::path::Path;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    #[structopt(short = "w", long = "weight", required = true)]
    weightfile: String,
    /// output dimention (>= 1)
    #[structopt(short = "d", long = "dim")]
    dim: Option<usize>,
    #[structopt(short = "i", long = "in")]
    gradin: Option<String>,
    #[structopt(short = "o", long = "out")]
    gradout: Option<String>,
    /// learning rate
    #[structopt(long = "lr", default_value = "0.01")]
    lr: f32,
    /// L2 regularization
    #[structopt(long = "l2", default_value = "0.0")]
    l2: f32,
}

fn main() {
    let opt = Opts::from_args();

    let mut x = matrix::read();
    // concat bias 1
    for i in 0..x.len() {
        x[i].push(1.0);
    }
    let (h, w) = matrix::shape(&x);

    let mut weight = if Path::new(&opt.weightfile).exists() {
        matrix::read_from_file(&opt.weightfile)
    } else {
        if let Some(dim) = opt.dim {
            matrix::random((w, dim))
        } else {
            panic!("Undetermined dim. Give --dim or existence weight file.")
        }
    };
    let d = matrix::shape(&weight).1;

    if w != weight.len() {
        panic!(format!("Incompatible weight dimention; dim(input) = {}, dim(weight) = {} + bias",
                       w - 1,
                       weight.len() - 1));
    }

    let y: Matrix = matrix::dot(&x, &weight);
    matrix::write(&y);

    if let Some(gradin_file) = opt.gradin {
        let gy = matrix::read_from_file(&gradin_file);

        if let Some(gradout_file) = opt.gradout {
            let gx: Matrix = (0..h).map(|i|
                (0..w).map(|j|
                    (0..d).map(|k| weight[j][k] * gy[i][k]).sum()
                ).collect()
            ).collect();
            matrix::write_to_file(&gx, &gradout_file);
        }
        // update weights
        let gw: Matrix = (0..w).map(|j|
            (0..d).map(|k|
                (0..h).map(|i| x[i][j] * gy[i][k]).sum()
            ).collect()
        ).collect();

        // L2 regularization
        if opt.l2 > 0.0 {
            let wr = 1.0 - opt.lr * opt.l2;
            for j in 0..w {
                for k in 0..d {
                    weight[j][k] *= wr;
                }
            }

        }
        // min Loss
        for j in 0..w {
            for k in 0..d {
                weight[j][k] -= gw[j][k] * opt.lr;
            }
        }
        matrix::write_to_file(&weight, &opt.weightfile);

    }
}
