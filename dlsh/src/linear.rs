mod data;
use data::matrix;
use data::matrix::Matrix;

fn main() {
    let x = matrix::read();
    let y: Matrix = x.iter().map(|row| {
        vec![row.iter().sum()]
    }).collect();
    matrix::write(&y);
}
