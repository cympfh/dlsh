mod data;
use data::matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    /// label type ("binary" "category")
    labeltype: String,
    /// true data file
    truefile: String,
}

#[derive(PartialEq, Eq)]
enum Acc {
    Binary,
    Category,
}

use Acc::*;

fn argmax(xs: &Vec<f32>) -> usize {
    let mut k = 0;
    let mut xk = xs[0];
    for i in 1..xs.len() {
        if xs[i] > xk {
            xk = xs[i];
            k = i;
        }
    }
    k
}

impl Acc {
    fn check(&self, x: &Vec<f32>, y: &Vec<f32>) -> bool {
        match self {
            Binary => {
                let th = 0.5;
                (x[0] > th) == (y[0] > th)
            },
            Category => {
                argmax(&x) == argmax(&y)
            },
        }
    }
}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();
    let (h, w) = matrix::shape(&x);

    let y = matrix::read_from_file(&opt.truefile);
    if matrix::shape(&y) != (h, w) {
        panic!(format!(
            "Imcompatible shape. Can accept same shape matrices: shape(input) = {:?}, shape(true) = {:?}",
            matrix::shape(&x),
            matrix::shape(&y)));
    }

    let acc = match opt.labeltype.as_ref() {
        "binary" => Binary,
        "category" => Category,
        _ => {
            panic!(format!("Unknown label type: {}", opt.labeltype))
        },
    };

    let mut num_correct = 0;
    for i in 0..h {
        if acc.check(&x[i], &y[i]) {
            num_correct += 1;
        }
    }

    println!("{:.4}", (num_correct as f64) / (h as f64));

}

