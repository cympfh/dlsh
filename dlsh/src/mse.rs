mod data;
use data::matrix;
use data::matrix::Matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    #[structopt(short = "t", long = "true", required = true)]
    truefile: String
}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();
    let y = matrix::read_from_file(opt.truefile);

    let (h, w) = matrix::shape(&x);
    let z: Matrix = (0..h).map(|i|
        (0..w).map(|j| (x[i][j] - y[i][j]).powi(2)).collect()
    ).collect();

    matrix::write(&z);
}
