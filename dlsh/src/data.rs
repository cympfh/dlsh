#![allow(dead_code)]

use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, LineWriter};
use std::iter::FromIterator;
use std::ops::{Add, Div, Index, IndexMut, Mul};
use std::path::Path;

/// Trait Total can compare floats
/// Thanks to: https://qiita.com/hatoo@github/items/fa14ad36a1b568d14f3e
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
struct Total<T>(T);
impl<T> Total<T> {
    fn unwrap(self) -> T {
        self.0
    }
}
impl<T: PartialEq> Eq for Total<T> {}
impl<T: PartialOrd> Ord for Total<T> {
    fn cmp(&self, rhs: &Total<T>) -> std::cmp::Ordering {
        self.0.partial_cmp(&rhs.0).unwrap()
    }
}

/// Row is a row of a Matrix
#[derive(Debug, Clone)]
pub struct Row {
    pub data: Vec<f32>,
}

impl Row {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    pub fn random(n: usize, var: f32) -> Self {
        let normal = Normal::new(0.0, var).unwrap();
        let data = (0..n)
            .map(|_| normal.sample(&mut rand::thread_rng()))
            .collect();
        Self { data }
    }

    pub fn push(&mut self, x: f32) {
        self.data.push(x);
    }

    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .map(Total)
            .enumerate()
            .max_by_key(|item| item.1)
            .unwrap()
            .0
    }
}

impl Index<usize> for Row {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.data[i]
    }
}

impl IndexMut<usize> for Row {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.data[i]
    }
}

impl FromIterator<f32> for Row {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let data: Vec<f32> = Vec::from_iter(iter);
        Row { data }
    }
}

impl Add<&Row> for &Row {
    type Output = Row;
    fn add(self, other: &Row) -> Row {
        let n = self.len();
        let data = (0..n).map(|i| self[i] + other[i]).collect();
        Row::new(data)
    }
}

impl Div<f32> for &Row {
    type Output = Row;
    fn div(self, k: f32) -> Row {
        let n = self.len();
        let data = (0..n).map(|i| self[i] / k).collect();
        Row::new(data)
    }
}

/// Matrix is 2d-tensor
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Row>,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        Self {
            data: data.into_iter().map(Row::new).collect::<Vec<_>>(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.len(), self.data[0].len())
    }

    pub fn at(&self, i: usize, j: usize) -> f32 {
        self[i][j]
    }

    pub fn zeros(h: usize, w: usize) -> Self {
        Self {
            data: vec![Row::zeros(w); h],
        }
    }

    pub fn random(h: usize, w: usize, var: f32) -> Self {
        let mut data = vec![];
        for _ in 0..h {
            data.push(Row::random(w, var));
        }
        Self { data }
    }

    pub fn read() -> Self {
        let stdin = std::io::stdin();
        let mut line = String::new();
        let _ = stdin.read_line(&mut line);
        let shape: Vec<usize> = line
            .split_whitespace()
            .map(|token| token.parse::<usize>().ok().unwrap())
            .collect();
        let data: Vec<Vec<f32>> = (0..shape[0])
            .map(|_| {
                let mut line = String::new();
                let _ = stdin.read_line(&mut line);
                line.split_whitespace()
                    .map(|token| token.parse().ok().unwrap())
                    .collect()
            })
            .collect();
        Self::new(data)
    }

    pub fn read_from_file(path: &str) -> Self {
        if let Ok(file) = File::open(Path::new(&path)) {
            let buf = BufReader::new(file);
            let mut data: Vec<Vec<f32>> = vec![];
            for (i, line) in buf.lines().enumerate() {
                if i == 0 {
                    // ignore size header
                } else {
                    data.push(
                        line.unwrap()
                            .split_whitespace()
                            .map(|token| token.parse().ok().unwrap())
                            .collect(),
                    )
                }
            }
            Matrix::new(data)
        } else {
            panic!(format!("No such file: {}", path));
        }
    }

    pub fn write(&self) {
        let (h, w) = self.shape();
        println!("{} {}", h, w);
        for i in 0..h {
            for j in 0..w {
                if j == 0 {
                    print!("{}", self.at(i, j));
                } else {
                    print!(" {}", self.at(i, j));
                }
            }
            println!();
        }
    }

    pub fn write_to_file(&self, path: &str) {
        let file = File::create(path).unwrap();
        let mut file = LineWriter::new(file);
        let (h, w) = self.shape();
        let _ = file.write_all(format!("{} {}\n", h, w).as_bytes());
        for i in 0..h {
            for j in 0..w {
                if j == 0 {
                    let _ = file.write_all(format!("{}", self.at(i, j)).as_bytes());
                } else {
                    let _ = file.write_all(format!(" {}", self.at(i, j)).as_bytes());
                }
            }
            let _ = file.write_all(b"\n");
        }
    }
}

impl Index<usize> for Matrix {
    type Output = Row;
    fn index(&self, i: usize) -> &Row {
        &self.data[i]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, i: usize) -> &mut Row {
        &mut self.data[i]
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Matrix {
        let (h, w) = self.shape();
        let data = (0..h)
            .map(|i| (0..w).map(|j| self.at(i, j) + other.at(i, j)).collect())
            .collect();
        Matrix::new(data)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Matrix {
        let (h, _) = self.shape();
        let (w, v) = other.shape();
        let data = (0..h)
            .map(|i| {
                (0..v)
                    .map(|k| (0..w).map(|j| self.data[i][j] * other.data[j][k]).sum())
                    .collect()
            })
            .collect();
        Matrix::new(data)
    }
}

impl FromIterator<Row> for Matrix {
    fn from_iter<I: IntoIterator<Item = Row>>(iter: I) -> Self {
        let data: Vec<Row> = Vec::from_iter(iter);
        Matrix { data }
    }
}
