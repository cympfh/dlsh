mod data;
use data::{Matrix, Row};

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

impl Acc {
    fn check(&self, x: &Row, y: &Row) -> bool {
        match self {
            Acc::Binary => {
                let th = 0.5;
                (x[0] > th) == (y[0] > th)
            }
            Acc::Category => x.argmax() == y.argmax(),
        }
    }
}

fn main() {
    let opt = Opts::from_args();

    let x = Matrix::read();
    let (h, w) = x.shape();

    let y = Matrix::read_from_file(&opt.truefile);
    if y.shape() != (h, w) {
        panic!(format!(
            "Imcompatible shape. Can accept same shape matrices: shape(input) = {:?}, shape(true) = {:?}",
            x.shape(), y.shape()));
    }

    let acc = match opt.labeltype.as_ref() {
        "binary" => Acc::Binary,
        "category" => Acc::Category,
        _ => panic!(format!("Unknown label type: {}", opt.labeltype)),
    };

    let mut num_correct = 0;
    for i in 0..h {
        if acc.check(&x[i], &y[i]) {
            num_correct += 1;
        }
    }

    println!("{:.4}", (num_correct as f64) / (h as f64));
}
