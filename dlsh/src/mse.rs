mod data;
use data::matrix;
use data::matrix::Matrix;

extern crate structopt;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "mse")]
struct Opts {
    #[structopt(short = "t", long = "true", required = true)]
    truefile: String,
    #[structopt(short = "o", long = "out")]
    gradout: Option<String>,
}

fn forward(x: f32, y: f32) -> f32 {
    (x - y).powi(2)
}

fn backward(x: f32, y: f32) -> f32 {
    x - y
}

fn main() {
    let opt = Opts::from_args();

    let x = matrix::read();
    let y = matrix::read_from_file(&opt.truefile);

    let (h, w) = matrix::shape(&x);
    let z: Matrix = (0..h).map(|i|
        (0..w).map(|j|
            forward(x[i][j], y[i][j])
        ).collect()
    ).collect();

    matrix::write(&z);

    if let Some(gradout_file) = opt.gradout {
        let g: Matrix = (0..h).map(|i|
            (0..w).map(|j|
                backward(x[i][j], y[i][j])
            ).collect()
        ).collect();
        matrix::write_to_file(&g, &gradout_file);
    }

}
