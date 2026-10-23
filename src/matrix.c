#include "matrix.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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

	for (unsigned row = 0; row < order; row++) {
		for (unsigned col = 0; col < order; col++) {
			float sum = 0.0f;
			for (unsigned k = 0; k < order; k++) {
				 sum += a->data[order * row + k] * b->data[order * k + col];
			}
			result->data[order * row + col] = sum;
		}
	}

	return result;
}

int mat_dump(const struct mat *mat, const char *path) {
	assert(mat != NULL && mat->data != NULL);
	assert(path != NULL);

	FILE *f = fopen(path, "wb");
	if (!f) {
		return -1;
	}

	uint64_t order = mat->order;
	size_t count = mat->order * mat->order;
	if (fwrite(&order, sizeof(order), 1, f) != 1 ||
	    fwrite(mat->data, sizeof(float), count, f) != count) {
		fclose(f);
		return -1;
	}

	return fclose(f) == 0 ? 0 : -1;
}

void mat_free(struct mat *mat) {
	free(mat->data);
	free(mat);
}
