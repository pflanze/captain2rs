#!/usr/bin/env python3

from cProfile import Profile
from pstats import SortKey, Stats

import timeit

import numpy as np
import sparse

from captain2rs import (dispersal_distances_threshold_rs,
                        DispersalDistancesThreshold)


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


rng = np.random.default_rng(42)

#A = rng.random((800, 800), dtype=np.float32)
A = np.array([
    [0, 9, 2, 0, 0, 0, 3, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 0]
], dtype=np.float32)

#B = np.random.rand(100, 100, 400, 400)
dumping_dist = pp("rust", mytime("rust", lambda: dispersal_distances_threshold_rs(A.shape[0], 2.3, 3)))
B = pp("dumping_dist", mytime("dumping_dist", lambda: dumping_dist ** (1 / 2.3)))  # something

def method_einsum():
    return pp("einsum", sparse.einsum("ij,ijnm->nm", A, B, dtype=np.float32).todense())

def method_inlined():
    ddt = DispersalDistancesThreshold(2.3, 3).map(lambda x: x ** (1 / 2.3));
    return pp("inlined", ddt.apply(A))

def method_tensordot():
    return np.tensordot(A, B, axes=([0, 1], [0, 1]))

def method_broadcasting():
    return (A[:, :, None, None] * B).sum(axis=(0, 1))

def profiling():
    einsum = mytime("method_einsum", method_einsum)
    inlined = mytime("method_inlined", method_inlined)
    print(einsum / inlined)
    print(einsum == inlined)
    # tensordot = mytime("method_tensordot", method_tensordot)
    # print(f"Tensordot:     {tensordot}")
    # broadcasting = mytime("method_broadcasting", method_broadcasting)
    # print(f"Broadcasting:  {broadcasting}")
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
    
