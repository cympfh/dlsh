mod data;
use data::matrix::{read, write};

fn main() {
    let x = read();
    let y: Vec<Vec<f32>> = x.iter().map(|row| {
        row.iter().map(|&v|
            1.0 / (1.0 + f32::exp(-v))
        ).collect()
    }).collect();
    write(&y);
}
