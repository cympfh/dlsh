use std::env;

mod data;
use data::{Matrix, Row};

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

#[derive(Debug)]
enum Activation {
    Sigmoid,
    Relu,
    Softmax,
}

/// clipped (-20, 80)
fn exp(x: f32) -> f32 {
    if x > 80.0 {
        80.0_f32.exp()
    } else if x < -20.0 {
        (-20.0_f32).exp()
    } else {
        x.exp()
    }
}

impl Activation {
    fn summary(&self, xs: &Row) -> f32 {
        match self {
            Activation::Sigmoid => 0.0,
            Activation::Relu => 0.0,
            Activation::Softmax => xs.data.iter().map(|&x| exp(x)).sum(),
        }
    }
    fn forward(&self, x: f32, sum: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + exp(-x)),
            Activation::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Softmax => exp(x) / sum,
        }
    }
    fn backward(&self, x: f32, y: f32, gy: f32) -> f32 {
        match self {
            Activation::Sigmoid => gy * y * (1.0 - y),
            Activation::Relu => {
                if x > 0.0 {
                    gy
                } else {
                    0.0
                }
            }
            Activation::Softmax => gy * y * (1.0 - y),
        }
    }
}

fn main() {
    let opt = Opts::from_args();
    let nan_debug = env::var("NAN_DEBUG").is_ok();

    let x = Matrix::read();
    let (h, w) = x.shape();

    let act = match opt.acttype.as_ref() {
        "sigmoid" => Activation::Sigmoid,
        "relu" => Activation::Relu,
        "softmax" => Activation::Softmax,
        _ => panic!(format!("Unknown activation type: {}", opt.acttype)),
    };

    if nan_debug {
        let num: usize = x
            .data
            .iter()
            .map(|row| row.data.iter().filter(|val| val.is_nan()).count())
            .sum();
        if num > 0 {
            eprintln!(
                "[Activation {}] NaN detected from Input ({} / {:?})",
                opt.acttype,
                num,
                (h, w)
            );
            panic!();
        }
    }

    let y: Matrix = (0..h)
        .map(|i| {
            let sum = act.summary(&x[i]);
            (0..w).map(|j| act.forward(x[i][j], sum)).collect()
        })
        .collect();
    y.write();

    if nan_debug {
        for i in 0..h {
            for j in 0..w {
                if y[i][j].is_nan() {
                    eprintln!(
                        "[Activation {}] NaN detected from Output; forward({}, {}) = NaN",
                        opt.acttype,
                        x[i][j],
                        act.summary(&x[i])
                    );
                    eprintln!("[Activation {}] x[] = {:?}", opt.acttype, &x[i]);
                    panic!();
                }
            }
        }
    }

    if let Some(gradin_file) = opt.gradin {
        let gy = Matrix::read_from_file(&gradin_file);
        if let Some(gradout_file) = opt.gradout {
            let gx: Matrix = (0..h)
                .map(|i| {
                    (0..w)
                        .map(|j| act.backward(x[i][j], y[i][j], gy[i][j]))
                        .collect()
                })
                .collect();
            gx.write_to_file(&gradout_file);
        }
    }
}
