# Dependency notices

The default CPU build uses Kokkos 4.5.01, Kokkos Kernels 4.5.01 and Eigen 3.4.0.
Their license texts and Eigen attribution are retained here and installed under
`share/fqo/licenses`. Eigen's attribution file comes from the Ubuntu 22.04
`libeigen3-dev` package used for local validation; it also lists optional Eigen
modules that this optimizer does not use.

Sources:

- https://github.com/kokkos/kokkos/tree/4.5.01
- https://github.com/kokkos/kokkos-kernels/tree/4.5.01
- https://gitlab.com/libeigen/eigen/-/tree/3.4.0

Site installations may use different dependency versions. Their own license
and notice files remain authoritative for those builds. The optimizer's own
license is in the repository's root `LICENSE`.
