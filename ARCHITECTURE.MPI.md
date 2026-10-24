# MPI block matrix multiplication

This document describes `src/mpi.c`, which computes `C = A · B` for two
square `n × n` matrices. It splits the product into block tasks and hands
them to worker processes through a **master/worker task farm**: workers pull
new tasks as they finish old ones.

```
mpirun -np P ./build/src/mpi -n ORDER [-s BLOCK_SIZE]
```

- `-n` is the matrix order `n`.
- `-s` is the block size `b` (default 4). It must divide `n`.
- `P` must be at least 2: one master and at least one worker.

## 1. Process model

`mpirun -np P` starts **P independent copies** of the same program. Each is
an OS process with its own address space, and nothing is shared. Data moves
between processes only through explicit MPI messages. The copies differ only
in their **rank**, an integer `0 … P-1` returned by `MPI_Comm_rank`.

```
                 mpirun -np 4 ./mpi -n 8 -s 4
                              │
       ┌──────────────┬───────┴──────┬──────────────┐
       ▼              ▼              ▼              ▼
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ rank 0  │    │ rank 1  │    │ rank 2  │    │ rank 3  │
  │ master  │    │ worker  │    │ worker  │    │ worker  │
  │         │    │         │    │         │    │         │
  │ A, B, C │    │ task    │    │ task    │    │ task    │
  │ (full)  │    │ buffer  │    │ buffer  │    │ buffer  │
  └─────────┘    └─────────┘    └─────────┘    └─────────┘
       ▲              ▲              ▲              ▲
       └──────────────┴──── messages ┴──────────────┘
                   (MPI_COMM_WORLD)
```

Every rank runs `main` from the top:

1. `MPI_Init`.
2. `parse_args`. Every rank gets the same `argv`, so **every rank knows `n`
   and `b` without being told**. Workers use them to size their buffers.
3. `MPI_Comm_rank` / `MPI_Comm_size`. If `P < 2`, rank 0 prints the error
   and all ranks exit cleanly.
4. The roles split on `rank == 0`:
   - **Rank 0 (master)** generates `A` and `B`, runs `distribute_blocks`,
     assembles `C`, times the parallel part, and checks the result against
     the serial `mat_mult`.
   - **Ranks 1 … P-1 (workers)** run `worker`: receive a task, compute,
     send back the result, and repeat until told to stop.
5. `MPI_Finalize`.

## 2. Data decomposition

With `q = n / b`, each matrix is a `q × q` grid of `b × b` blocks. Each
output block is a sum over a block row of `A` and a block column of `B`:

```
C[i][j] = Σ  A[i][k] · B[k][j]        k = 0 … q-1
          k
```

Example: `n = 8`, `b = 2`, so `q = 4`. Computing `C[1][2]`:

```
          A                    B                    C
  ┌────┬────┬────┬────┐  ┌────┬────┬────┬────┐  ┌────┬────┬────┬────┐
  │    │    │    │    │  │    │    │░░░░│    │  │    │    │    │    │
  ├────┼────┼────┼────┤  ├────┼────┼────┼────┤  ├────┼────┼────┼────┤
  │░░░░│░░░░│░░░░│░░░░│  │    │    │░░░░│    │  │    │    │████│    │
  ├────┼────┼────┼────┤ ·├────┼────┼────┼────┤ =├────┼────┼────┼────┤
  │    │    │    │    │  │    │    │░░░░│    │  │    │    │    │    │
  ├────┼────┼────┼────┤  ├────┼────┼────┼────┤  ├────┼────┼────┼────┤
  │    │    │    │    │  │    │    │░░░░│    │  │    │    │    │    │
  └────┴────┴────┴────┘  └────┴────┴────┴────┘  └────┴────┴────┴────┘
     block row i = 1       block column j = 2      block (1, 2)
```

**One task computes one output block `C[i][j]`.** There are `q²` tasks.
They are independent, so they can be computed in any order, by any worker.

Tasks are numbered `t = 0 … q²-1` in row-major order, so a single `int`
identifies one:

```
i = t / q        j = t % q
```

### Why one task per `C[i][j]` and not per block pair

The alternative would be one task per product `A[i][k] · B[k][j]`. Compare
the two (master traffic in floats):

| task granularity            | messages | traffic through master | sum over k |
|-----------------------------|----------|------------------------|------------|
| pair `A[i][k] · B[k][j]`    | `~2q³`   | `3 n² q`               | master     |
| output block `C[i][j]`      | `~2q²`   | `2 n² q + n²`          | worker     |

With one task per output block there are `q` times fewer messages, the master
never adds up partial sums, and each worker returns exactly one finished
block per task.

## 3. Memory layout and packing

Matrices are stored row-major in `struct mat` (`src/matrix.h`):
element `(r, c)` is `data[r * n + c]`.

A `b × b` block is **not contiguous** in that layout. Its rows are `n`
floats apart:

```
 data (n = 8, b = 2), block (1, 2) highlighted:

 row 0  │ .  .  .  .  .  .  .  . │
 row 1  │ .  .  .  .  .  .  .  . │
 row 2  │ .  .  .  .  X  X  .  . │  ← starts at data[(1*2)*8 + 2*2] = data[20]
 row 3  │ .  .  .  .  X  X  .  . │  ← next row of the block is 8 floats later
 row 4  │ .  .  .  .  .  .  .  . │
          ...
```

`MPI_Send(buf, count, MPI_FLOAT, …)` sends `count` floats that sit next to
each other in memory, so blocks are first **packed** (copied) into contiguous
buffers. The matrix library provides:

| function             | purpose                                                         |
|----------------------|-----------------------------------------------------------------|
| `mat_get_block`      | copy block `(row, col)` out of a matrix into a contiguous buffer |
| `mat_set_block`      | inverse: copy a contiguous buffer into block `(row, col)`       |
| `mat_block_mult_add` | `c += a · b` on contiguous `b × b` blocks (i-k-j loop order)    |
| `mat_zero`           | allocate a zeroed matrix (used for `C`)                         |

### Task message layout

`pack_task` puts everything for `C[i][j]` into **one** buffer of
`2 · q · b²` floats, so a task travels as a single message:

```
 task buffer
 ┌─────────┬─────────┬─────┬───────────┬─────────┬─────────┬─────┬───────────┐
 │ A[i][0] │ A[i][1] │ ... │ A[i][q-1] │ B[0][j] │ B[1][j] │ ... │ B[q-1][j] │
 └─────────┴─────────┴─────┴───────────┴─────────┴─────────┴─────┴───────────┘
 ▲                                     ▲
 a_row = task                          b_col = task + q·b²
 (each cell is b² floats)
```

The worker reads it in place:

```c
memset(c, 0, b*b * sizeof(float));
for (k = 0; k < q; k++)
	mat_block_mult_add(a_row + k*b*b, b_col + k*b*b, c, b);
```

The result message is just the `b²` floats of `c`.

## 4. Communication protocol

### Message types

All messages use `MPI_COMM_WORLD` and `MPI_FLOAT`. The **tag** tells the
receiver what kind of message it is.

| tag          | direction       | payload                   | meaning                     |
|--------------|-----------------|---------------------------|-----------------------------|
| `TAG_TASK`   | master → worker | `2 · q · b²` floats       | compute this `C[i][j]`      |
| `TAG_RESULT` | worker → master | `b²` floats               | here is the finished block  |
| `TAG_STOP`   | master → worker | 0 floats (`NULL`, 0)      | no more work, leave loop    |

Note that the payload does **not** say which `C[i][j]` it is:

- The master remembers what it handed out in `assigned[worker] = t`. Each
  worker has at most one task at a time, so this table is enough to place
  every result.
- The worker doesn't need to know `i` and `j` at all.

### Master: `distribute_blocks`

```
 next = 0, busy = 0

 ┌─ prime ───────────────────────────────────────────────┐
 │ for w in 1 … P-1:                                     │
 │     if next < q²:  send TASK(next) to w               │
 │                    assigned[w] = next++ ; busy++      │
 │     else:          send STOP to w   (spare worker)    │
 └───────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌─ serve ───────────────────────────────────────────────┐
 │ while busy > 0:                                       │
 │     recv RESULT from ANY_SOURCE  → status             │
 │     w = status.MPI_SOURCE                             │
 │     place result at block assigned[w] of C            │
 │     if next < q²:  send TASK(next) to w               │
 │                    assigned[w] = next++               │
 │     else:          send STOP to w ; busy--            │
 └───────────────────────────────────────────────────────┘
                          │
                          ▼
                   return C (complete)
```

The core of the design: **a result also means "I'm free"**. The master never
picks which worker gets the next task. It replies to whichever worker
reported first (`MPI_ANY_SOURCE`). Fast workers come back sooner and so take
more tasks, which is load balancing with no extra logic.

### Worker: `worker`

```
 ┌──────────────────────────────────────────────┐
 │ loop:                                        │
 │     recv from 0, ANY_TAG  → status           │◄──────┐
 │     if status.MPI_TAG == TAG_STOP: break     │       │
 │     c = 0                                    │       │
 │     c += Σ_k a_row[k] · b_col[k]             │       │
 │     send RESULT(c) to 0                      │───────┘
 └──────────────────────────────────────────────┘
```

### Example timeline

`P = 3` (two workers), `q = 2` (four tasks), with worker 2 slower than
worker 1:

```
   master (0)                 worker 1                  worker 2
       │                          │                         │
       │── TASK t=0 ─────────────►│                         │
       │── TASK t=1 ──────────────┼────────────────────────►│
       │                          │ compute C[0][0]         │ compute C[0][1]
       │◄──────────── RESULT ─────│                         │   (slow)
       │  place at t=0            │                         │
       │── TASK t=2 ─────────────►│                         │
       │                          │ compute C[1][0]         │
       │◄──────────── RESULT ─────│                         │
       │  place at t=2            │                         │
       │── TASK t=3 ─────────────►│                         │
       │                          │ compute C[1][1]         │
       │◄─────────────────────────┼────────── RESULT ───────│
       │  place at t=1            │                         │
       │── STOP ──────────────────┼────────────────────────►│ exit loop
       │◄──────────── RESULT ─────│                         │
       │  place at t=3            │                         │
       │── STOP ─────────────────►│ exit loop               │
       │                          │                         │
   busy == 0, C complete
```

Worker 1 ended up with three tasks and worker 2 with one. Round-robin would
have given each worker two, and the master would have waited for the slow
one.

### Termination

- Every worker receives **exactly one** `TAG_STOP`. Workers that get a first
  task receive it after their last result. Spare workers (more workers than
  tasks) receive it during priming.
- `busy` counts the workers that hold a task. The master's loop ends when it
  reaches 0, which is exactly when every outstanding result has been
  collected.
- So every rank reaches `MPI_Finalize`, and no rank is left blocked in a
  receive.

## 5. MPI semantics this design relies on

- **Blocking send/receive.**
  - `MPI_Recv` returns once the message is in the buffer.
  - `MPI_Send` returns once its buffer is safe to reuse. That is why the
    master packs every task into the **same** `task` buffer. (This would
    change with non-blocking `MPI_Isend`.)
- **Message matching.**
  - A receive matches on `(source, tag, communicator)`.
  - The master uses `MPI_ANY_SOURCE` with a fixed tag. The worker uses a
    fixed source with `MPI_ANY_TAG`.
  - `MPI_Status` reports what actually matched.
- **Short receives are legal.** The `count` in a receive is the maximum it
  accepts, so the zero-length `TAG_STOP` matches a receive posted for a full
  task.
- **Ordering.** MPI keeps messages in order only between one sender and one
  receiver. Results from different workers arrive in any order, which is why
  results are identified through `assigned[]` and not by arrival order.
- **Failure.** `xmalloc` calls `MPI_Abort` on allocation failure. That kills
  the whole job. Calling `exit()` instead would leave the other ranks blocked
  forever.

## 6. Costs and limitations

Let `n` be the order, `b` the block size, `q = n / b` and `W = P - 1` the
number of workers.

| quantity                    | value                                         |
|-----------------------------|-----------------------------------------------|
| tasks                       | `q²`                                          |
| messages                    | `2q² + W` (task + result each, plus stops)    |
| floats sent by master       | `q² · 2qb² = 2 n² q`                          |
| floats received by master   | `q² · b² = n²`                                |
| flops per task              | `2 · q · b³ = 2 n b²`                         |
| flops per float transferred | `≈ b`                                         |

What this means in practice:

- **The block size is a trade-off.**
  - A small `b` gives many small messages. Message latency dominates, and
    each float sent pays for little computation (`≈ b` flops).
  - A large `b` gives few, large tasks. Near the end some workers sit idle
    while others finish, and there are fewer tasks than workers when
    `q² < W`.
- **The master is a serial bottleneck.**
  - Every byte goes in and out of rank 0, one message at a time.
  - It also does all the packing.
  - The total input traffic, `2 n² q`, grows with `q`: each block of `A` is
    sent `q` times, once for every `j`.
  - Adding workers stops helping once the master can't keep them busy.
- **Communication and computation don't overlap.** A worker sits idle from
  sending its result until its next task arrives. One fix is to prefetch: keep
  two tasks outstanding per worker.
- **Memory.** Rank 0 holds full copies of `A`, `B` and `C`. Each worker holds
  only `(2q + 1) · b²` floats.
- **The master does no computation.** One of the `P` processes only
  coordinates.

For comparison, Cannon's algorithm or SUMMA avoid the central bottleneck.
There every rank owns a fixed tile and exchanges tiles with its neighbours,
so the traffic is spread across all processes.

## 7. Verification

After the parallel phase, rank 0 computes `mat_mult(A, B)` serially and
prints the maximum absolute difference. Workers add the `k` terms in the same
order as the serial loop, so the result is normally identical (difference
`0`). The serial check is `O(n³)` on one core and dominates the run time for
large `n`.
