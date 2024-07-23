# Ultra Scale Fractale Based optimization

This repository contains Implementations of certain Fractale based decomposition optimization algorithms at an ultra scale level moving from intranode with one gpu to internode level.

It is done using Kokkos for the intranode with one gpu scale level and using MPI + Kokkos for both levels intranode with multiple gpus and internode.

We use both the device and the host so you are free to build kokkos with your prefered back-ends for both, we have run these scripts using OpenMp as host back-end and Cuda as device back-end.

## References

T. Firmin and E.-G. Talbi, "Massively Parallel Asynchronous Fractal Optimization," in 2023 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), St. Petersburg, FL, USA, 2023, pp. 930-938, doi: 10.1109/IPDPSW59300.2023.00151.

Keywords: Distributed processing; Conferences; Software algorithms; Search problems; Linear programming; Fractals; Software; Asynchronous metaheuristic; Continuous optimization; Fractals; Hierarchical decomposition.
