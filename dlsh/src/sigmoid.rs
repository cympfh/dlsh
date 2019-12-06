mod data;
use data::matrix;
use data::matrix::Matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    #[structopt(short = "i", long = "in")]
    gradin: Option<String>,
    #[structopt(short = "o", long = "out")]
    gradout: Option<String>,
}

fn forward(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();
    let (h, w) = matrix::shape(&x);

    let y: Matrix = (0..h).map(|i|
        (0..w).map(|j|
            forward(x[i][j])
        ).collect()
    ).collect();
    matrix::write(&y);

    if let Some(gradin_file) = opt.gradin {
        if let Some(gradout_file) = opt.gradout {
            let gy = matrix::read_from_file(&gradin_file);
            let gx: Matrix = (0..h).map(|i|
                (0..w).map(|j|
                    gy[i][j] * y[i][j] * (1.0 - y[i][j])
                ).collect()
            ).collect();
            matrix::write_to_file(&gx, &gradout_file);
        }
    }
}
