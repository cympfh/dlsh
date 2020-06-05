mod data;
use data::{Matrix, Row};

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
    #[structopt(short = "s", long = "summary", default_value = "nothing")]
    summary: String,
}

const EPS: f32 = 1e-9;

/// clipped [eps, inf)
fn log(x: f32) -> f32 {
    (if x < EPS { EPS } else { x }).ln()
}

#[derive(Debug, PartialEq, Eq)]
enum Loss {
    MSE,
    KL,
}

impl Loss {
    fn forward(&self, x: f32, y: f32) -> f32 {
        match self {
            Loss::MSE => (x - y).powi(2),
            Loss::KL => x * (log(x) - log(y)),
        }
    }
    fn backward(&self, x: f32, y: f32) -> f32 {
        match self {
            Loss::MSE => x - y,
            Loss::KL => log(x) - log(y) + 1.0,
        }
    }
}

#[derive(Debug)]
enum LossSummary {
    Nothing,
    Average,
}

impl LossSummary {
    fn call(&self, x: &Matrix) {
        match self {
            LossSummary::Nothing => x.write(),
            LossSummary::Average => {
                let (h, w) = x.shape();
                let mut sum = Row::zeros(w);
                for row in x.data.iter() {
                    sum = &sum + row;
                }
                sum = &sum / h as f32;
                let avg = Matrix { data: vec![sum] };
                avg.write()
            }
        }
    }
}

fn main() {
    let opt = Opts::from_args();

    let loss = match opt.losstype.as_ref() {
        "mse" => Loss::MSE,
        "kl" => Loss::KL,
        _ => panic!(format!("Unknown loss type: {}", opt.losstype)),
    };

    let summary = match opt.summary.as_ref() {
        "Average" | "average" => LossSummary::Average,
        _ => LossSummary::Nothing,
    };

    let x = Matrix::read();
    let (h, w) = x.shape();

    let y = Matrix::read_from_file(&opt.truefile);
    if y.shape() != (h, w) {
        panic!(format!(
            "[{:?}] Imcompatible shape. Can accept same shape matrices: shape(input) = {:?}, shape(true) = {:?}",
            loss,
            x.shape(),
            y.shape()));
    }

    let z: Matrix = (0..h)
        .map(|i| (0..w).map(|j| loss.forward(x[i][j], y[i][j])).collect())
        .collect();

    // Report average
    summary.call(&z);

    if let Some(gradout_file) = opt.gradout {
        let g: Matrix = (0..h)
            .map(|i| (0..w).map(|j| loss.backward(x[i][j], y[i][j])).collect())
            .collect();
        g.write_to_file(&gradout_file);
    }
}
