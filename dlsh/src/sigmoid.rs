mod data;
use data::matrix;
use data::matrix::Matrix;

fn main() {
    let x = matrix::read();
    let y: Matrix = x.iter().map(|row| {
        row.iter().map(|&v|
            1.0 / (1.0 + f32::exp(-v))
        ).collect()
    }).collect();
    matrix::write(&y);
}
