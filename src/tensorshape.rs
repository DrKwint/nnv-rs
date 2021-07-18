use std::ops::Index;

pub struct TensorShape {
	dims: Vec<Option<usize>>,
}

impl TensorShape {
	pub fn new(dims: Vec<Option<usize>>) -> Self {
		Self { dims }
	}

	pub fn is_fully_defined(&self) -> bool {
		self.dims.iter().all(|x| x.is_some())
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

impl Index<usize> for TensorShape {
	type Output = Option<usize>;

	fn index(&self, idx: usize) -> &Option<usize> {
		&self.dims[idx]
	}
}

impl From<Vec<usize>> for TensorShape {
	fn from(v: Vec<usize>) -> Self {
		Self {
			dims: v.into_iter().map(|x| Some(x)).collect(),
		}
	}
}
