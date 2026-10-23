#ifndef MATMUL_MATRIX_H
#define MATMUL_MATRIX_H

#include <stddef.h>

struct mat {
	size_t order;
	float *data;
};

struct mat *mat_rand(unsigned order);
struct mat *mat_mult(struct mat *a, struct mat *b);

int mat_dump(const struct mat *mat, const char *path);

void mat_free(struct mat *mat);

#endif  // MATMUL_MATRIX_H
