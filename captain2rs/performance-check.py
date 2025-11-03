#!/usr/bin/env python3

from cProfile import Profile
from pstats import SortKey, Stats

import timeit

import numpy as np
import sparse

from captain2rs import (dispersal_distances_threshold_rs,
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


rng = np.random.default_rng(42)

A = rng.random((800, 800), dtype=floattype)

dumping_dist = pp_mytime("rust", lambda: dispersal_distances_threshold_rs(A.shape[0], 2.3, 3))
B = pp_mytime("dumping_dist", lambda: dumping_dist ** (1 / 2.3))

def bench_einsum():
    return sparse.einsum("ij,ijnm->nm", A, B, dtype=floattype).todense()

def bench_inlined():
    ddt = DispersalDistancesThreshold(2.3, 3).map(lambda x: x ** (1 / 2.3));
    return ddt.apply(A)

def bench_tensordot():
    return np.tensordot(A, B, axes=([0, 1], [0, 1]))

def bench_broadcasting():
    return (A[:, :, None, None] * B).sum(axis=(0, 1))

def profiling():
    einsum = pp_mytime("einsum", bench_einsum)
    inlined = pp_mytime("inlined", bench_inlined)
    print(einsum / inlined)
    print(einsum == inlined)
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
    
