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
    #[structopt(short = "d", long = "dim", required = true)]
    dim: usize,
}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();

    // load weights
    let w = if Path::new(&opt.weightfile).exists() {
        matrix::read_from_file(&opt.weightfile)
    } else {
        let w = matrix::random((x[0].len(), opt.dim));
        matrix::write_to_file(&w, &opt.weightfile);
        w
    };

    let y: Matrix = matrix::dot(&x, &w);
    matrix::write(&y);

}
