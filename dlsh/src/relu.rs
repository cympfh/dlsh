mod data;
use data::matrix::{read, write};

fn main() {
    let x = read();
    let y: Vec<Vec<f32>> = x.iter().map(|row| {
        row.iter().map(|&v|
            if v > 0.0 {
                v
            } else {
                0.0
            }
        ).collect()
    }).collect();
    write(&y);
}

