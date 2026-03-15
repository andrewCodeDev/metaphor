//! Shape
//!
//! Fixed-capacity shape abstraction for tensor dimensions.
//! Maximum 8 dimensions. Buffer is 1-initialized so unused
//! dimensions don't affect size calculations (product).
//!

module metaphor::tensor::shape;

const usz CAPACITY = 8;

// Indices — 0-initialized position tracking

struct Indices
{
	ulong[CAPACITY] buffer;
	usz len;
}

const Indices EMPTY_INDICES = { .buffer = { [0..7] = 0 }, .len = 0 };

fn void Indices.set(&self, usz i, ulong value)
{
	self.buffer[i] = value;
}

fn ulong Indices.get(&self, usz i)
{
	return self.buffer[i];
}

fn ulong[] Indices.slice(&self)
{
	return self.buffer[:self.len];
}

// Strides — 1-initialized stride computation

struct Strides
{
	ulong[CAPACITY] buffer;
	usz len;
}

/// Calculate inner product between indices and strides to create offset.
fn ulong Strides.pos_to_offset(&self, ulong[] indices)
{
	ulong offset = 0;
	usz n = indices.len < self.len ? indices.len : self.len;
	for (usz i = 0; i < n; i++)
	{
		offset += indices[i] * self.buffer[i];
	}
	return offset;
}

/// Convert linear offset to multi-dimensional position.
fn Indices Strides.offset_to_pos(&self, ulong offset)
{
	Indices pos = EMPTY_INDICES;
	pos.len = self.len;
	ulong remaining = offset;
	for (usz i = 0; i < self.len; i++)
	{
		pos.set(i, remaining / self.buffer[i]);
		remaining %= self.buffer[i];
	}
	return pos;
}

fn ulong[] Strides.slice(&self)
{
	return self.buffer[:self.len];
}

fn ulong Strides.get(&self, usz i)
{
	return self.buffer[i];
}

// Shape — 1-initialized dimension buffer

struct Shape
{
	ulong[CAPACITY] buffer;
	usz len;
}

/// Empty shape with all dimensions = 1.
const Shape EMPTY_SHAPE = { .buffer = { [0..7] = 1 }, .len = 0 };

/// Initialize shape from a slice of dimension values.
fn Shape shape_init(ulong[] values)
{
	assert(values.len <= CAPACITY);
	Shape s = EMPTY_SHAPE;
	s.len = values.len;
	for (usz i = 0; i < values.len; i++)
	{
		s.buffer[i] = values[i];
	}
	return s;
}

fn void Shape.set(&self, usz i, ulong value)
{
	self.buffer[i] = value;
}

fn ulong Shape.get(&self, usz i)
{
	return self.buffer[i];
}

fn ulong Shape.last(&self)
{
	assert(self.len > 0);
	return self.buffer[self.len - 1];
}

fn ulong[] Shape.slice(&self)
{
	return self.buffer[:self.len];
}

fn ulong[] Shape.head(&self, usz amount)
{
	return self.buffer[:amount];
}

fn ulong[] Shape.tail(&self, usz amount)
{
	return self.buffer[self.len - amount .. self.len];
}

/// Crop shape[lhs:-rhs]
fn ulong[] Shape.crop(&self, usz lhs, usz rhs)
{
	if (lhs < self.len && rhs < self.len)
	{
		usz end = self.len - rhs;
		if (end < lhs) { end = lhs; }
		return self.buffer[lhs..end];
	}
	return {};
}

/// Total number of elements (product of all dimensions).
fn ulong Shape.size(&self)
{
	ulong result = 1;
	for (usz i = 0; i < CAPACITY; i++)
	{
		result *= self.buffer[i];
	}
	return result;
}

fn void Shape.append(&self, ulong value)
{
	assert(self.len < CAPACITY);
	self.set(self.len, value);
	self.len++;
}

fn void Shape.remove(&self, usz i)
{
	assert(i < self.len);
	// Shift elements left
	for (usz j = i; j < self.len - 1; j++)
	{
		self.buffer[j] = self.buffer[j + 1];
	}
	self.buffer[self.len - 1] = 1;
	self.len--;
}

/// Count dimensions with a given value.
fn usz Shape.count(&self, ulong value)
{
	usz c = 0;
	for (usz i = 0; i < CAPACITY; i++)
	{
		if (self.buffer[i] == value) { c++; }
	}
	return c;
}

/// Check if two shapes are element-wise equal.
fn bool Shape.equal(&self, Shape* other)
{
	for (usz i = 0; i < CAPACITY; i++)
	{
		if (self.buffer[i] != other.buffer[i]) { return false; }
	}
	return true;
}

/// Check if two shapes are broadcast-compatible.
fn bool Shape.compatible(&self, Shape other)
{
	usz dims = self.len > other.len ? self.len : other.len;
	for (usz i = 0; i < dims; i++)
	{
		ulong dim_a = i < self.len ? self.get(self.len - 1 - i) : 1;
		ulong dim_b = i < other.len ? other.get(other.len - 1 - i) : 1;
		if (dim_a != dim_b && dim_a != 1 && dim_b != 1) { return false; }
	}
	return true;
}

/// Compute broadcast shape. Returns false if not broadcastable.
fn bool Shape.broadcast(&self, Shape* other, Shape* result)
{
	if (self.equal(other))
	{
		*result = *self;
		return true;
	}
	usz dims = self.len > other.len ? self.len : other.len;
	*result = EMPTY_SHAPE;
	result.len = dims;
	for (usz i = 0; i < dims; i++)
	{
		ulong dim_a = i < self.len ? self.get(self.len - 1 - i) : 1;
		ulong dim_b = i < other.len ? other.get(other.len - 1 - i) : 1;
		if (dim_a != dim_b && dim_a != 1 && dim_b != 1) { return false; }
		result.set(dims - 1 - i, dim_a > dim_b ? dim_a : dim_b);
	}
	return true;
}

/// Number of "real" (non-1) dimensions.
fn usz Shape.realdims(&self)
{
	usz c = 0;
	for (usz i = 0; i < CAPACITY; i++)
	{
		if (self.buffer[i] != 1) { c++; }
	}
	return c;
}

/// Squeeze out dimensions of size 1.
fn Shape Shape.squeeze(&self)
{
	Shape tmp = *self;
	// Can't squeeze below a single dimension — [1] stays [1], not scalar []
	if (self.len <= 1)
	{
		return *self;
	}
	usz j = 0;
	for (usz i = 0; i < self.len; i++)
	{
		if (self.get(i) != 1)
		{
			tmp.set(j, self.get(i));
			j++;
		}
	}
	// Fill remaining with 1
	for (usz k = j; k < CAPACITY; k++)
	{
		tmp.buffer[k] = 1;
	}
	tmp.len = j;
	return tmp;
}

/// Add a leading dimension of 1.
fn Shape Shape.unsqueeze(&self)
{
	assert(self.len < CAPACITY);
	Shape tmp = EMPTY_SHAPE;
	tmp.len = self.len + 1;
	tmp.buffer[0] = 1;
	for (usz i = 0; i < self.len; i++)
	{
		tmp.buffer[i + 1] = self.buffer[i];
	}
	return tmp;
}

/// Compute strides for this shape (row-major).
fn Strides Shape.strides(&self)
{
	Strides s;
	s.len = self.len;

	if (self.len == 0)
	{
		// Scalar: stride 0 so in[i*0] = in[0]
		for (usz i = 0; i < CAPACITY; i++) { s.buffer[i] = 0; }
		return s;
	}

	if (self.len == 1)
	{
		s.buffer[0] = 1;
		for (usz i = 1; i < CAPACITY; i++) { s.buffer[i] = 1; }
		return s;
	}

	// General case: suffix product scan
	// strides[len-1] = 1
	// strides[i] = product of shape[i+1..len]
	for (usz i = 0; i < CAPACITY; i++) { s.buffer[i] = 1; }

	s.buffer[self.len - 1] = 1;
	usz i = self.len - 1;
	while (i > 0)
	{
		i--;
		s.buffer[i] = s.buffer[i + 1] * self.buffer[i + 1];
	}
	return s;
}

/// Aligned mismatch — find first dimension where shapes differ,
/// accounting for rank differences.
struct MismatchResult
{
	usz a_pos;
	usz b_pos;
}

fn bool Shape.mismatch(&self, Shape other, MismatchResult* result)
{
	usz a_shift = self.len > other.len ? self.len - other.len : 0;
	usz b_shift = other.len > self.len ? other.len - self.len : 0;

	ulong[] a_crop = self.crop(a_shift, 0);
	ulong[] b_crop = other.crop(b_shift, 0);

	usz compare_len = a_crop.len < b_crop.len ? a_crop.len : b_crop.len;
	for (usz i = 0; i < compare_len; i++)
	{
		if (a_crop[i] != b_crop[i])
		{
			result.a_pos = a_shift + i;
			result.b_pos = b_shift + i;
			return true;
		}
	}
	return false; // no mismatch
}

/// Merge multiple shape slices into one shape.
fn Shape shape_merge(ulong[][] slices)
{
	Shape tmp = EMPTY_SHAPE;
	for (usz s = 0; s < slices.len; s++)
	{
		for (usz i = 0; i < slices[s].len; i++)
		{
			tmp.buffer[tmp.len] = slices[s][i];
			tmp.len++;
		}
	}
	return tmp;
}
