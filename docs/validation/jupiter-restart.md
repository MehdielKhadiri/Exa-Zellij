# JUPITER checkpoint and restart tests

Both FBD and FBO passed checkpoint and restart tests on JUPITER on
24 October 2026. The tests used Booster nodes with four NVIDIA GH200 GPUs
and four MPI processes per node. Optimizer kernels and the Rastrigin
objective ran on the GPUs.

The reference tests used commit
[`a975949`](https://github.com/MehdielKhadiri/Exa-Zellij/tree/a975949dae5652b5b66cdec868b34c9f97f9c297).
All 23 saved source files match that commit after accounting for line endings.
The executable's SHA-256 checksum is
`aa2c85525a99ba59cb278dcede469fed6cf7bd47f5870c065c1b7a366b8958c5`.
A checksum identifies the exact file used for a run.

## Configuration

The build used GCC 14.3.0, CUDA 13, Kokkos 4.7.03, Kokkos Kernels 4.7.01
and ParaStationMPI 5.13.0-1-mt. The tests used 128-dimensional Rastrigin,
a maximum depth of four and distributed initial sampling. FBD's settings
determined its work; FBO used an evaluation budget.

## Recovery with the same number of processes

| Booster nodes | MPI processes | Job | Result |
| --- | --- | --- | --- |
| 1 | 4 | 2487563 | Passed |
| 2 | 8 | 2487625 | Passed |
| 10 | 40 | 2487771 | Passed |
| 25 | 100 | 2487897 | Passed |
| 50 | 200 | 2488067 | Passed |

At 10, 25 and 50 nodes, the tests killed a process after saving a checkpoint,
then restarted the program. Both families recovered the final point and value
of the uninterrupted reference run, using relative tolerance `1e-8` and
absolute tolerance `1e-7`. At 50 nodes, further tests damaged primary files
and checked recovery from an earlier checkpoint or backup copies.

## Recovery with a different number of processes

Job 2488067 tested **100 to 200 processes and 200 to 100 processes** for
both families, with three seeds. All twelve restarted runs completed. Each
restart loaded the saved depth and redistributed the evaluated points and
their values. The result file records the saved point counts and returned
objective values for each run.

## Node aggregation

A separate version added node aggregation: each node leader writes its
processes' records into one file. On 24 October, job 2489712 passed the same
100-to-200 and 200-to-100 process restart checks. The checks confirmed that
redistribution preserved the saved point/value records exactly. Smaller
tests in job 2488970 checked restart between four and eight processes.

Across the aggregation tests, 86 comparisons found identical checkpoint
record bytes and 116 checks confirmed file counts and sizes. At 50 nodes,
aggregation reduced the number of data files, including backups, from
400 to 100. This extension has its own executable checksum,
`03f4442a32844c64a365c20be689104696419ba978924b57a4f045839c2d1079`;
it is separate from the reference source linked above.

## Saved results

[jupiter-restart-results.json](jupiter-restart-results.json) contains the job
identifiers, source and executable checksums, restart results and verification
totals extracted from the saved test records. It also records the checksums
of the original job summaries. Personal storage paths are omitted.
