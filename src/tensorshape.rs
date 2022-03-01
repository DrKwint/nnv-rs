#![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::ops::Index;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShape {
    dims: Vec<Option<usize>>,
}

impl TensorShape {
    /// # Panics
    pub fn new(dims: Vec<Option<usize>>) -> Self {
        assert!(dims.iter().any(Option::is_some));
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> usize {
        self.dims.iter().fold(1, |a, b| a * b.unwrap_or(1))
    }

    pub fn is_fully_defined(&self) -> bool {
        self.dims.iter().all(Option::is_some)
    }

    pub fn as_defined_slice(&self) -> Option<Vec<usize>> {
        let slice: Vec<usize> = self.dims.iter().filter_map(|&x| x).collect();
        if slice.len() == self.rank() {
            Some(slice)
        } else {
            None
        }
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        if self.dims == vec![None] {
            return true;
        }
        if self.dims.len() != other.dims.len() {
            return false;
        }
        self.dims
            .iter()
            .zip(other.dims.iter())
            .all(|(x, y)| match (x, y) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            })
    }
}

impl From<TensorShape> for Vec<Option<usize>> {
    fn from(ts: TensorShape) -> Self {
        ts.dims
    }
}

impl Index<isize> for TensorShape {
    type Output = Option<usize>;

    fn index(&self, mut idx: isize) -> &Option<usize> {
        if idx < 0 {
            idx += self.dims.len() as isize;
        }
        debug_assert!(idx >= 0, "idx {} < 0", idx);
        &self.dims[idx as usize]
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(v: Vec<usize>) -> Self {
        Self {
            dims: v.into_iter().map(Some).collect(),
        }
    }
}

impl Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let strs = self
            .dims
            .iter()
            .map(|opt| opt.map_or("None".to_owned(), |x| x.to_string()))
            .collect::<Vec<String>>();
        write!(f, "({})", strs.join(", "))
    }
}
