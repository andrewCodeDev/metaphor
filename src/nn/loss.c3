//! Loss Functions
//!
//! Standard loss functions for neural network training.
//! All losses return scalar tensors.
//!

module metaphor::nn::loss;
import metaphor::tensor;
import metaphor::nn::functional;

// MSE Loss

/// Mean Squared Error loss.
///   L = mean((pred - target)^2)
fn tensor::Tensor? mse_loss(tensor::Tensor pred, tensor::Tensor target)
{
	tensor::Tensor diff = pred.sub(target)!;
	tensor::Tensor squared = diff.sqr()!;
	return functional::mean(squared);
}

// Cross-Entropy Loss

/// Cross-entropy loss (expects probabilities, not logits).
///   L = -mean(sum(target * log(pred), axis=-1))
///
/// pred:   [batch, classes] — predicted probabilities (after softmax)
/// target: [batch, classes] — one-hot encoded targets
fn tensor::Tensor? cross_entropy(tensor::Tensor pred, tensor::Tensor target)
{
	tensor::Tensor eps = pred.scalar_like(1e-7)!;
	tensor::Tensor clamped = pred.max(eps)!;
	tensor::Tensor log_pred = clamped.log()!;
	tensor::Tensor prod = target.mul(log_pred)!;

	usz rank = pred.shape().len;
	functional::ReduceExpandPatterns pat = functional::patterns_for_dim(rank, rank - 1)!;
	tensor::Tensor sum_classes = prod.reduce(pat.reduce_pat())!;
	tensor::Tensor loss = functional::mean(sum_classes)!;
	return loss.neg();
}

/// Binary cross-entropy loss.
///   L = -mean(target * log(pred) + (1 - target) * log(1 - pred))
fn tensor::Tensor? binary_cross_entropy(tensor::Tensor pred, tensor::Tensor target)
{
	tensor::Tensor eps = pred.scalar_like(1e-7)!;
	tensor::Tensor one = pred.scalar_like(1.0)!;
	tensor::Tensor one_minus_eps = pred.scalar_like(1.0 - 1e-7)!;

	// Clamp pred to [eps, 1-eps]
	tensor::Tensor clamped = pred.max(eps)!;
	clamped = clamped.min(one_minus_eps)!;

	// target * log(pred) + (1 - target) * log(1 - pred)
	tensor::Tensor log_pred = clamped.log()!;
	tensor::Tensor one_minus_pred = one.sub(clamped)!;
	tensor::Tensor log_one_minus = one_minus_pred.log()!;
	tensor::Tensor one_minus_target = one.sub(target)!;

	tensor::Tensor term1 = target.mul(log_pred)!;
	tensor::Tensor term2 = one_minus_target.mul(log_one_minus)!;

	tensor::Tensor sum_terms = term1.add(term2)!;
	tensor::Tensor loss = functional::mean(sum_terms)!;
	return loss.neg();
}

/// Softmax cross-entropy loss (numerically stable).
///   L = -mean(sum(target * log_softmax(logits), axis=-1))
///
/// logits: [batch, classes] — raw logits (before softmax)
/// target: [batch, classes] — one-hot encoded targets
/// weight: [batch] — optional per-position weights (pass NULL_TENSOR to disable)
///   When provided: L = -sum(weight * sum(target * log_softmax(logits), axis=-1)) / sum(weight)
fn tensor::Tensor? softmax_cross_entropy(
	tensor::Tensor logits, tensor::Tensor target,
	tensor::Tensor weight = tensor::NULL_TENSOR)
{
	usz rank = logits.shape().len;
	tensor::Tensor log_probs = functional::log_softmax(logits, rank - 1)!;
	tensor::Tensor prod = target.mul(log_probs)!;

	functional::ReduceExpandPatterns pat = functional::patterns_for_dim(rank, rank - 1)!;
	tensor::Tensor sum_classes = prod.reduce(pat.reduce_pat())!;
	tensor::Tensor neg_ce = sum_classes.neg()!;

	if (weight._core != null)
	{
		return weighted_mean(neg_ce, weight);
	}

	return functional::mean(neg_ce);
}

// Weighted Loss Variants

/// Weighted MSE loss.
///   L = sum(weight * (pred - target)^2) / sum(weight)
fn tensor::Tensor? weighted_mse(
	tensor::Tensor pred, tensor::Tensor target, tensor::Tensor weight)
{
	tensor::Tensor diff = pred.sub(target)!;
	tensor::Tensor squared = diff.sqr()!;
	tensor::Tensor weighted = squared.mul(weight)!;
	return weighted_mean(weighted, weight);
}

/// Weighted cross-entropy loss.
fn tensor::Tensor? weighted_cross_entropy(
	tensor::Tensor pred, tensor::Tensor target, tensor::Tensor weight)
{
	tensor::Tensor eps = pred.scalar_like(1e-7)!;
	tensor::Tensor clamped = pred.max(eps)!;
	tensor::Tensor log_pred = clamped.log()!;
	tensor::Tensor prod = target.mul(log_pred)!;

	usz wrank = pred.shape().len;
	functional::ReduceExpandPatterns wpat = functional::patterns_for_dim(wrank, wrank - 1)!;
	tensor::Tensor sum_classes = prod.reduce(wpat.reduce_pat())!;
	tensor::Tensor neg_ce = sum_classes.neg()!;
	return weighted_mean(neg_ce, weight);
}

// Helpers

/// Weighted mean: sum(values * weights) / sum(weights)
fn tensor::Tensor? weighted_mean(tensor::Tensor values, tensor::Tensor weights)
{
	tensor::Tensor weighted = values.mul(weights)!;
	tensor::Tensor sum_values = functional::sum_all(weighted)!;
	tensor::Tensor sum_weights = functional::sum_all(weights)!;
	return sum_values.div(sum_weights);
}
