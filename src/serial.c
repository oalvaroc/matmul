#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>

#include "matrix.h"

static void help(char *prog) {
	printf("usage: %s OPTIONS\n\n", prog);
	printf("Multiply two square matrices.\n\n");
	printf("Options:\n");
	printf("    -n, --order   Square matrix order\n");
	printf("    -h, --help    Show this message and exit\n");
}

static void parse_args(int argc, char **argv, int *order) {
	static struct option opts[] = {
		{"help",  no_argument,       NULL, 'h'},
		{"order", required_argument, NULL, 'n'},
		{NULL,    0,                 NULL,  0 }
	};

	int opt;
	while ((opt = getopt_long(argc, argv, "hn:", opts, NULL)) > 0) {
		switch (opt) {
		case 'h':
			help(argv[0]);
			exit(EXIT_SUCCESS);
		case 'n':
			*order = atoi(optarg);
			if (*order < 2) {
				printf("matrix order must be a number greater than 2\n");
				exit(EXIT_FAILURE);
			}
			break;
		case '?':
			break;
		default:
			abort();
		}
	}

	if (*order == 0) {
		printf("missing required option: -n, --order\n");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	if (argc <= 1) {
		help(argv[0]);
		return 0;
	}

	int order = 0;
	parse_args(argc, argv, &order);

	struct mat *a = mat_rand(order);
	struct mat *b = mat_rand(order);

	struct timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);

	struct mat *c = mat_mult(a, b);

	struct timespec end;
	clock_gettime(CLOCK_MONOTONIC, &end);

	double delta_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	printf("order=%d elapsed=%.6lfs\n", order, delta_s);

	mat_dump(a, "a.mat");
	mat_dump(b, "b.mat");
	mat_dump(c, "result.mat");

	mat_free(a);
	mat_free(b);
	mat_free(c);
	return 0;
}
