#![allow(dead_code)]

pub mod matrix {

    use std::fs::File;
    use std::path::Path;
    use std::io::BufReader;
    use std::io::prelude::*;

    pub type Matrix = Vec<Vec<f32>>;

    pub fn shape(x: &Matrix) -> (usize, usize) {
        (x.len(), x[0].len())
    }

    pub fn read() -> Matrix {
        let stdin = std::io::stdin();
        let mut line = String::new();
        let _ = stdin.read_line(&mut line);
        let shape: Vec<usize> = line.split_whitespace().map(|token| token.parse::<usize>().ok().unwrap()).collect();
        return (0..shape[0]).map(|_| {
            let mut line = String::new();
            let _ = stdin.read_line(&mut line);
            line.split_whitespace().map(|token| token.parse().ok().unwrap()).collect()
        }).collect()
    }

    pub fn read_from_file(path: String) -> Matrix {
        let file = File::open(Path::new(&path)).unwrap();
        let buf = BufReader::new(file);
        let mut x: Matrix = vec![];
        for (i, line) in buf.lines().enumerate() {
            if i == 0 {
                // ignore size header
            } else {
                x.push(
                    line.unwrap().split_whitespace().map(|token| token.parse().ok().unwrap()).collect()
                )
            }
        }
        return x
    }

    pub fn write(y: &Matrix) {
        let (h, w) = shape(&y);
        println!("{} {}", h, w);
        for i in 0..h {
            for j in 0..w {
                if j == 0 {
                    print!("{}", y[i][j]);
                } else {
                    print!(" {}", y[i][j]);
                }
            }
            println!("");
        }
    }
}
