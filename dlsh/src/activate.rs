mod data;
use data::matrix;
use data::matrix::Matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    /// activation type (e.g. sigmoid, relu)
    acttype: String,
    #[structopt(short = "i", long = "in")]
    gradin: Option<String>,
    #[structopt(short = "o", long = "out")]
    gradout: Option<String>,
}

enum Activation {
    Sigmoid,
    Relu,
    Softmax,
}

use Activation::*;

impl Activation {
    fn summary(&self, xs: &Vec<f32>) -> f32 {
        match self {
            Sigmoid => 0.0,
            Relu => 0.0,
            Softmax => {
                xs.iter().map(|&x| x.exp()).sum()
            },
        }
    }
    fn forward(&self, x: f32, sum: f32) -> f32 {
        match self {
            Sigmoid => {
                1.0 / (1.0 + f32::exp(-x))
            },
            Relu => {
                if x > 0.0 { x } else { 0.0 }
            },
            Softmax => {
                x.exp() / sum
            },
        }
    }
    fn backward(&self, x: f32, y: f32, gy: f32) -> f32 {
        match self {
            Activation::Sigmoid => {
                gy * y * (1.0 - y)
            },
            Activation::Relu => {
                if x > 0.0 {
                    gy
                } else {
                    0.0
                }
            },
            Activation::Softmax => {
                gy * y * (1.0 - y)
            },
        }
    }
}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();
    let (h, w) = matrix::shape(&x);

    let act = match opt.acttype.as_ref() {
        "sigmoid" => Activation::Sigmoid,
        "relu" => Activation::Relu,
        "softmax" => Activation::Softmax,
        _ => {
            panic!(format!("Unknown activation type: {}", opt.acttype))
        }
    };

    let y: Matrix = (0..h).map(|i| {
        let sum = act.summary(&x[i]);
        (0..w).map(|j|
            act.forward(x[i][j], sum)
        ).collect()
    }).collect();
    matrix::write(&y);

    if let Some(gradin_file) = opt.gradin {
        let gy = matrix::read_from_file(&gradin_file);
        if let Some(gradout_file) = opt.gradout {
            let gx: Matrix = (0..h).map(|i|
                (0..w).map(|j|
                    act.backward(x[i][j], y[i][j], gy[i][j])
                ).collect()
            ).collect();
            matrix::write_to_file(&gx, &gradout_file);
        }
    }
}
