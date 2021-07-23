use std::ops::Index;

pub struct TensorShape {
	dims: Vec<Option<usize>>,
}

impl TensorShape {
	pub fn new(dims: Vec<Option<usize>>) -> Self {
		Self { dims }
	}

	pub fn rank(&self) -> usize {
		self.dims.len()
	}

	pub fn is_fully_defined(&self) -> bool {
		self.dims.iter().all(|x| x.is_some())
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

impl Index<isize> for TensorShape {
	type Output = Option<usize>;

	fn index(&self, mut idx: isize) -> &Option<usize> {
		if idx < 0 {
			idx = self.dims.len() as isize - idx;
		}
		&self.dims[idx as usize]
	}
}

impl From<Vec<usize>> for TensorShape {
	fn from(v: Vec<usize>) -> Self {
		Self {
			dims: v.into_iter().map(|x| Some(x)).collect(),
		}
	}
}
