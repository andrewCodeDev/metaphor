//! Streaming Dataset Reader
//!
//! Loads MNIST data from IDX files and organizes into batches with
//! 256-byte aligned strides. Supports one-hot label encoding.
//!
//! Usage:
//!   DatasetReader reader = dataset::init_mnist("data", true, 64)!!;
//!   defer reader.free();
//!   while (reader.next()) |batch| {
//!       // memcpy batch.images/batch.labels into input tensors
//!   }
//!   reader.reset();
//!

module metaphor::utils::dataset;
import metaphor::utils::idx;
import std::core::string;
import std::io;

// Constants

/// Memory alignment for optimal GPU performance (256 bytes).
const usz ALIGNMENT = 256;

/// Round up byte count to alignment boundary.
fn usz align_up(usz size)
{
	return ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
}

// Batch — returned by iterator

struct Batch
{
	float[] images;  // [batch_size * 784] flattened
	float[] labels;  // [batch_size * 10] one-hot flattened
	usz size;        // actual batch size
}

// DatasetReader

faultdef DATASET_LOAD_FAILED;

struct DatasetReader
{
	usz batch_size;
	usz sample_count;

	// Aligned strides (in elements) — includes padding for 256-byte alignment
	usz image_stride;  // elements per batch of images (>= batch_size * 784)
	usz label_stride;  // elements per batch of labels (>= batch_size * 10)

	// All data in memory (host, aligned layout)
	float* all_images;
	usz images_len;
	float* all_labels;
	usz labels_len;

	// Iteration state
	usz current_batch;
}

/// Load MNIST from IDX files in data_dir.
/// Set `training` to true for training set, false for test set.
/// Run download_data.sh first to download the data.
fn DatasetReader? init_mnist(String data_dir, bool training, usz batch_size)
{
	String images_file = training ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte";
	String labels_file = training ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte";
	String images_path = string::tformat("%s/%s", data_dir, images_file);
	String labels_path = string::tformat("%s/%s", data_dir, labels_file);

	String set_name = training ? "training" : "test";
	io::printfn("Loading MNIST %s set from %s...", set_name, data_dir);

	idx::IdxData images_idx = idx::read_file(images_path)!;
	defer images_idx.free();

	idx::IdxData labels_idx = idx::read_file(labels_path)!;
	defer labels_idx.free();

	usz sample_count = images_idx.count();
	io::printfn("Loaded %d samples", (int)sample_count);

	// Calculate aligned strides (in elements)
	usz image_batch_bytes = batch_size * 784 * float.sizeof;
	usz label_batch_bytes = batch_size * 10 * float.sizeof;
	usz image_stride = align_up(image_batch_bytes) / float.sizeof;
	usz label_stride = align_up(label_batch_bytes) / float.sizeof;

	usz n_batches = sample_count / batch_size;

	// Allocate with aligned layout
	usz images_len = n_batches * image_stride;
	usz labels_len = n_batches * label_stride;

	float* all_images = (float*)mem::calloc(images_len * float.sizeof);
	assert(all_images != null, "dataset: image allocation failed");

	float* all_labels = (float*)mem::calloc(labels_len * float.sizeof);
	assert(all_labels != null, "dataset: label allocation failed");

	// Copy data into aligned layout with one-hot labels
	for (usz sample_idx = 0; sample_idx < n_batches * batch_size; sample_idx++)
	{
		usz batch_idx = sample_idx / batch_size;
		usz within_batch = sample_idx % batch_size;

		// Copy image pixels (already normalized to 0-1 by idx reader)
		usz image_base = batch_idx * image_stride + within_batch * 784;
		float[] src_image = images_idx.get(sample_idx);
		for (usz i = 0; i < 784 && i < src_image.len; i++)
		{
			all_images[image_base + i] = src_image[i];
		}

		// Create one-hot label
		usz label_base = batch_idx * label_stride + within_batch * 10;
		// Labels are normalized ubytes: value * 255 gives original label
		usz label = (usz)(labels_idx.data[sample_idx] * 255.0 + 0.5);
		for (usz i = 0; i < 10; i++)
		{
			all_labels[label_base + i] = (i == label) ? 1.0 : 0.0;
		}
	}

	return {
		.batch_size = batch_size,
		.sample_count = sample_count,
		.image_stride = image_stride,
		.label_stride = label_stride,
		.all_images = all_images,
		.images_len = images_len,
		.all_labels = all_labels,
		.labels_len = labels_len,
		.current_batch = 0,
	};
}

/// Free all allocated memory.
fn void DatasetReader.free(&self)
{
	if (self.all_images != null)
	{
		mem::free(self.all_images);
		self.all_images = null;
	}
	if (self.all_labels != null)
	{
		mem::free(self.all_labels);
		self.all_labels = null;
	}
}

/// Get next batch, or null if epoch complete.
fn Batch* DatasetReader.next(&self)
{
	usz n_batches = self.sample_count / self.batch_size;
	if (self.current_batch >= n_batches) return null;

	usz img_start = self.current_batch * self.image_stride;
	usz lbl_start = self.current_batch * self.label_stride;

	self.current_batch++;

	// Return via static to avoid heap allocation per batch
	static Batch batch;
	batch = {
		.images = self.all_images[img_start .. img_start + self.batch_size * 784 - 1],
		.labels = self.all_labels[lbl_start .. lbl_start + self.batch_size * 10 - 1],
		.size = self.batch_size,
	};
	return &batch;
}

/// Reset for next epoch.
fn void DatasetReader.reset(&self)
{
	self.current_batch = 0;
}

/// Number of complete batches per epoch.
fn usz DatasetReader.num_batches(&self)
{
	return self.sample_count / self.batch_size;
}

/// Total samples in dataset.
fn usz DatasetReader.num_samples(&self)
{
	return self.sample_count;
}
