#include "matrix.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>


struct mat *mat_zero(unsigned order) {
	assert(order >= 2);

	struct mat *mat = calloc(1, sizeof(struct mat));
	if (!mat) {
		return NULL;
	}

	size_t size = order * order;
	mat->order = order;
	mat->data = calloc(size, sizeof(float));
	if (!mat->data) {
		free(mat);
		return NULL;
	}

	return mat;
}

struct mat *mat_rand(unsigned order) {
	assert(order >= 2);

	struct mat *mat = mat_zero(order);
	if (!mat) {
		return NULL;
	}

	for (unsigned row = 0; row < order; row++) {
		for (unsigned col = 0; col < order; col++) {
			mat->data[order * row + col] = (1.0f * rand()) / RAND_MAX;
		}
	}

	return mat;
}

struct mat *mat_mult(struct mat *a, struct mat *b) {
	assert(a != NULL && b != NULL);
	assert(a->order == b->order);
	assert(a->data != NULL && b->data != NULL);

	const size_t order = a->order;

	struct mat *result = mat_zero(order);
	if (!result) {
		return NULL;
	}

	// row-k-col order walks b and result row by row, which is cache friendly
	for (unsigned row = 0; row < order; row++) {
		for (unsigned k = 0; k < order; k++) {
			const float a_row_k = a->data[row * order + k];
			for (unsigned col = 0; col < order; col++) {
				 result->data[row * order + col] += a_row_k * b->data[k * order + col];
			}
		}
	}

	return result;
}

void mat_get_block(const struct mat *mat, size_t row, size_t col, size_t bs,
		   float *buf) {
	assert(mat != NULL && mat->data != NULL);
	assert(buf != NULL);
	assert(bs > 0 && mat->order % bs == 0);
	assert(row < mat->order / bs && col < mat->order / bs);

	const size_t order = mat->order;
	const float *src = mat->data + (row * bs) * order + col * bs;

	for (size_t bl_row = 0; bl_row < bs; bl_row++) {
		memcpy(buf + bl_row * bs, src + bl_row * order, bs * sizeof(float));
	}
}

// Inverse of mat_get_block: copy the contiguous bs x bs buf into block
// (row, col) of mat.
void mat_set_block(struct mat *mat, size_t row, size_t col, size_t bs,
		   const float *buf) {
	assert(mat != NULL && mat->data != NULL);
	assert(buf != NULL);
	assert(bs > 0 && mat->order % bs == 0);
	assert(row < mat->order / bs && col < mat->order / bs);

	const size_t order = mat->order;
	float *dst = mat->data + (row * bs) * order + col * bs;

	for (size_t bl_row = 0; bl_row < bs; bl_row++) {
		memcpy(dst + bl_row * order, buf + bl_row * bs, bs * sizeof(float));
	}
}

// c += a * b, where all three are contiguous bs x bs row-major blocks.
void mat_block_mult_add(const float *a, const float *b, float *c, size_t bs) {
	assert(a != NULL && b != NULL && c != NULL);

	// row-k-col order walks b and c row by row, which is cache friendly
	for (size_t row = 0; row < bs; row++) {
		for (size_t k = 0; k < bs; k++) {
			const float a_row_k = a[row * bs + k];
			for (size_t col = 0; col < bs; col++) {
				c[row * bs + col] += a_row_k * b[k * bs + col];
			}
		}
	}
}

// Largest relative error of an entry of mat against ref, |m - r| / |r|.
double mat_max_rel_err(const struct mat *mat, const struct mat *ref) {
	assert(mat != NULL && ref != NULL);
	assert(mat->order == ref->order);

	const size_t size = mat->order * mat->order;
	double max = 0;

	for (size_t i = 0; i < size; i++) {
		const double m = mat->data[i];
		const double r = ref->data[i];
		const double diff = fabs(m - r);
		const double rel = r != 0 ? diff / fabs(r) : (diff != 0 ? INFINITY : 0);
		if (rel > max) {
			max = rel;
		}
	}

	return max;
}

// Largest relative error per entry accepted between two products of this
// order computed in single precision with the sums in different orders.
//
// Each entry is a sum of order products, and every addition rounds with a
// relative error of up to FLT_EPSILON / 2, up or down almost at random. Like
// a random walk, these errors add up to about sqrt(order) * FLT_EPSILON;
// the factor 10 is a safety margin.
double mat_tolerance(size_t order) {
	return 10 * sqrt(order) * FLT_EPSILON;
}

void mat_free(struct mat *mat) {
	free(mat->data);
	free(mat);
}
