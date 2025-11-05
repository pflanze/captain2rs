#!/usr/bin/env python3

from cProfile import Profile
from pstats import SortKey, Stats

import timeit

import numpy as np
import sparse

from captain2rs import (dispersal_distances_threshold_rs, num_candidates_rs,
                        DispersalDistancesThreshold)

floattype=np.float64

def mytime(nam, proc):
    res=()
    def f():
        nonlocal res
        res=proc()
    t=timeit.timeit(f, number=1)
    print(f"t {nam}={t:.6f}")
    return res

def pp(nam, val):
    print(f"pp {nam}: {val}")
    return val

def pp_mytime(nam, proc):
    return pp(nam, mytime(nam, proc))


lambda_0 = np.array([2.3, 4.1, 0.9, 4, 2, 9, 6.4, 10, 0.3, 3.9, 4.1, 8.3], dtype=floattype)
n_species = len(lambda_0)
print(f"n_species={n_species}")

rng = np.random.default_rng(42)

h = np.array([rng.random((800, 800), dtype=floattype) for i in range(n_species)], dtype=floattype)

threshold=3

dumping_dist = pp_mytime("rust", lambda: dispersal_distances_threshold_rs(h[0].shape[0], 2.3, threshold))

def bench_einsum():
    return np.array(
          [
              sparse.einsum(
                  "ij,ijnm->nm",
                  h[i],
                  dumping_dist ** (1 / lambda_0[i]),
                  dtype=floattype
              ).todense()
              for i in range(n_species)
          ])

def bench_rust_python():
    ddt=DispersalDistancesThreshold(lambda_0=2.3, threshold=threshold)
    return np.array(
          [
              ddt.map(lambda x: x ** (1 / lambda_0[i])).apply(h[i])
              for i in range(n_species)
          ])

def bench_rust_parallel():
    return num_candidates_rs(
        n_species=n_species,
        lambda_0_init=2.3,
        threshold=threshold,
        lambda_0=lambda_0,
        h=h)

# def bench_tensordot():
#     return np.tensordot(A, B, axes=([0, 1], [0, 1]))

# def bench_broadcasting():
#     return (A[:, :, None, None] * B).sum(axis=(0, 1))

def profiling():
    rust_python = pp_mytime("rust_python", bench_rust_python)
    rust_parallel = pp_mytime("rust_parallel", bench_rust_parallel)
    print("rust_python == rust_parallel: ", rust_python == rust_parallel)
    einsum = pp_mytime("einsum", bench_einsum)
    print("einsum / rust_python: ", einsum / rust_python)
    print("einsum == rust_python: ", einsum == rust_python)
    # tensordot = pp_mytime("tensordot", bench_tensordot)
    # broadcasting = pp_mytime("broadcasting", bench_broadcasting)
    # print(einsum == tensordot)
    # print(einsum == broadcasting)
    # print(tensordot == broadcasting)
    print("")
    print("")


profiling()

with Profile() as profile:
    profiling()
    (
         Stats(profile)
         .strip_dirs()
         .sort_stats(SortKey.TIME)
         .print_stats(15)
    )


pp_mytime("rust_parallel", bench_rust_parallel)
pp_mytime("rust_parallel", bench_rust_parallel)
pp_mytime("rust_parallel", bench_rust_parallel)
pp_mytime("rust_parallel", bench_rust_parallel)
pp_mytime("rust_parallel", bench_rust_parallel)
