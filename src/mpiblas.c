#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include <mpi.h>
#include <openblas/cblas.h>

#include "matrix.h"

#define DEFAULT_BLOCK_SIZE 4
#define DEFAULT_SEED 1

// message tags: a label that lets the receiver tell kinds of messages apart
#define TAG_TASK   1
#define TAG_RESULT 2
#define TAG_STOP   3

static void help(char *prog) {
	printf("usage: %s OPTIONS\n\n", prog);
	printf("Multiply two square matrices.\n\n");
	printf("Options:\n");
	printf("    -n, --order   Square matrix order\n");
	printf("    -s, --bs      Block size, must divide the order (default=4)\n");
	printf("    -r, --seed    Seed for the matrix RNG (default=1)\n");
	printf("    -h, --help    Show this message and exit\n");
}

static int parse_args(int argc, char **argv, int *order, int *block_size,
		      unsigned *seed) {
	static struct option opts[] = {
		{"help",  no_argument,       NULL, 'h'},
		{"order", required_argument, NULL, 'n'},
		{"bs",    required_argument, NULL, 's'},
		{"seed",  required_argument, NULL, 'r'},
		{NULL,    0,                 NULL,  0 }
	};

	int opt;
	while ((opt = getopt_long(argc, argv, "hn:s:r:", opts, NULL)) > 0) {
		switch (opt) {
		case 'h':
			help(argv[0]);
			return EXIT_FAILURE;
		case 'n':
			*order = atoi(optarg);
			if (*order < 2) {
				printf("matrix order must be a number greater than 2\n");
				return EXIT_FAILURE;
			}
			break;
		case 's':
			*block_size = atoi(optarg);
			if (*block_size < 1) {
				printf("block size must be a number greater than 0\n");
				return EXIT_FAILURE;
			}
			break;
		case 'r':
			*seed = strtoul(optarg, NULL, 10);
			break;
		case '?':
			break;
		default:
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	if (*order == 0) {
		printf("missing required option: -n, --order\n");
		return EXIT_FAILURE;
	}
	if (*order % *block_size != 0) {
		printf("block size %d does not divide order %d\n", *block_size, *order);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

static void *xmalloc(size_t size) {
	void *p = calloc(1, size);
	if (!p) {
		// kills every rank, not only this one
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	return p;
}

// A task is everything needed to compute C[i][j] = sum_k A[i][k] * B[k][j]:
// the q blocks A[i][0..q-1] followed by the q blocks B[0..q-1][j], packed
// into one contiguous buffer so it travels in a single message.
static void pack_task(const struct mat *a, const struct mat *b, size_t i,
		      size_t j, size_t bs, float *task) {
	const size_t q = a->order / bs;
	const size_t blk = bs * bs;

	for (size_t k = 0; k < q; k++) {
		mat_get_block(a, i, k, bs, task + k * blk);
		mat_get_block(b, k, j, bs, task + (q + k) * blk);
	}
}

// Task t computes C[t / q][t % q]. Packs it and sends it to worker w.
static void send_task(const struct mat *a, const struct mat *b, size_t bs,
		      int t, int w, float *task, int task_count) {
	const size_t q = a->order / bs;

	pack_task(a, b, t / q, t % q, bs, task);
	// MPI_Send returns once task can be reused, so one buffer is enough
	MPI_Send(task, task_count, MPI_FLOAT, w, TAG_TASK, MPI_COMM_WORLD);
}

// A message with no data: only the tag matters.
static void send_stop(int w) {
	MPI_Send(NULL, 0, MPI_FLOAT, w, TAG_STOP, MPI_COMM_WORLD);
}

static struct mat *distributed_mat_mult(struct mat *a, struct mat *b, size_t bs,
				     int nprocs) {
	const size_t q = a->order / bs;
	const int ntasks = q * q;
	const int task_count = 2 * q * bs * bs;
	const int result_count = bs * bs;

	float *task = xmalloc(task_count * sizeof(float));
	float *result = xmalloc(result_count * sizeof(float));
	// which task each worker is busy with, and how many it has done
	int *assigned = xmalloc(nprocs * sizeof(int));
	int *done = xmalloc(nprocs * sizeof(int));

	struct mat *c = mat_zero(a->order);

	// give every worker its first task; if there are more workers than
	// tasks, the spare ones are stopped right away
	int next = 0;
	int busy = 0;
	for (int w = 1; w < nprocs; w++) {
		if (next < ntasks) {
			send_task(a, b, bs, next, w, task, task_count);
			assigned[w] = next++;
			busy++;
		} else {
			send_stop(w);
		}
	}

	// a result also means "I'm free": answer with more work or a stop
	while (busy > 0) {
		MPI_Status status;
		MPI_Recv(result, result_count, MPI_FLOAT, MPI_ANY_SOURCE,
			 TAG_RESULT, MPI_COMM_WORLD, &status);

		const int w = status.MPI_SOURCE;
		const int t = assigned[w];
		mat_set_block(c, t / q, t % q, bs, result);
		done[w]++;

		if (next < ntasks) {
			send_task(a, b, bs, next, w, task, task_count);
			assigned[w] = next++;
		} else {
			send_stop(w);
			busy--;
		}
	}

	for (int w = 1; w < nprocs; w++) {
		printf("rank %d computed %d of %d blocks\n", w, done[w], ntasks);
	}

	free(done);
	free(assigned);
	free(result);
	free(task);
	return c;
}

static void worker(int order, size_t bs) {
	// the worker parsed the same argv, so it can size its buffers without
	// being told
	const size_t q = order / bs;
	const size_t blk = bs * bs;
	const int task_count = 2 * q * blk;

	float *task = xmalloc(task_count * sizeof(float));
	float *c = xmalloc(blk * sizeof(float));

	for (;;) {
		// accept any tag from the master, then look at which one it was
		MPI_Status status;
		MPI_Recv(task, task_count, MPI_FLOAT, 0, MPI_ANY_TAG,
			 MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == TAG_STOP) {
			break;
		}

		const float *a_row = task;           // A[i][0..q-1]
		const float *b_col = task + q * blk; // B[0..q-1][j]

		// c accumulates, so clear what the previous task left in it
		memset(c, 0, blk * sizeof(float));
		for (size_t k = 0; k < q; k++) {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, bs, 1,
		                    a_row + k * blk, bs, b_col + k * blk, bs, 1, c, bs);
		}

		MPI_Send(c, blk, MPI_FLOAT, 0, TAG_RESULT, MPI_COMM_WORLD);
	}

	free(c);
	free(task);
}

int main(int argc, char **argv) {
	// on failure the default error handler aborts before returning
	MPI_Init(&argc, &argv);

	// every rank receives the same argv, so every rank knows the order
	int order = 0;
	int block_size = DEFAULT_BLOCK_SIZE;
	unsigned seed = DEFAULT_SEED;
	if (parse_args(argc, argv, &order, &block_size, &seed) != EXIT_SUCCESS) {
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2) {
		if (rank == 0) {
			fprintf(stderr, "need at least 2 processes (1 master + workers)\n");
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	if (rank == 0) {
		// only the master generates matrices, in the same order as the
		// serial program, so the same seed gives the same A and B
		srand(seed);
		struct mat *a = mat_rand(order);
		struct mat *b = mat_rand(order);

		double start = MPI_Wtime();
		struct mat *c = distributed_mat_mult(a, b, block_size, size);
		double elapsed = MPI_Wtime() - start;

		printf("order=%d bs=%d procs=%d elapsed=%.6fs\n", order,
		       block_size, size, elapsed);

		mat_free(c);
		mat_free(a);
		mat_free(b);
	} else {
		worker(order, block_size);
	}

	MPI_Finalize();
	return 0;
}
