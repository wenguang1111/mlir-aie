# frenet_planner/frenet_planner.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# ─── FISS+ Frenet Planner – NPU IRON Design ────────────────────────────────
#
# Maps the brute-force Frenet Optimal Planner (FOP) baseline onto the 5-column
# AMD Ryzen AI NPU.  Each of the 5 columns runs an independent Worker that
# processes N_TRAJS_PER_COL=25 trajectories (5 col × 25 = 125 total) through
# all four pipeline stages inside a single combined kernel:
#
#   Stage 1 – Quintic/quartic polynomial generation
#   Stage 2 – Frenet → Cartesian via CubicSpline2D reference path
#   Stage 3 – Kinematic constraint filter (speed / acceleration limits)
#   Stage 4 – Collision check (6-point SAT ego vs 4-point obstacle polygons)
#
# Host sends (per planning cycle):
#   - 5 × params_buf  (96 bf16 each): initial FrenetState + 25 end states
#   - 5 × spline_buf  (928 bf16):     reference-path spline coefficients
#   - 5 × obs_buf     (3232 bf16):    obstacle polygon array
#
# Host receives:
#   - 5 × result_buf  (256 bf16):     best trajectory per worker
#
# The host Python/C++ wrapper post-processes the 5 results and selects the
# globally lowest-cost collision-free trajectory.
#
# Buffer size constants (must match frenet_planner_kernel.cc):
#   PARAMS_SIZE  =  96
#   SPLINE_SIZE  = 928
#   OBS_SIZE     = 3232
#   RESULT_SIZE  = 256
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path

# ── Compile-time constants (mirror frenet_planner_kernel.cc) ─────────────────
N_COLS          = 5
PARAMS_SIZE     = 96    # bf16 elements per params buffer
SPLINE_SIZE     = 928   # bf16 elements per spline buffer
OBS_SIZE        = 3232  # bf16 elements per obstacle buffer
RESULT_SIZE     = 256   # bf16 elements per result buffer

# ── Buffer types ──────────────────────────────────────────────────────────────
params_ty  = np.ndarray[(PARAMS_SIZE,),  np.dtype[bfloat16]]
spline_ty  = np.ndarray[(SPLINE_SIZE,),  np.dtype[bfloat16]]
obs_ty     = np.ndarray[(OBS_SIZE,),     np.dtype[bfloat16]]
result_ty  = np.ndarray[(RESULT_SIZE,),  np.dtype[bfloat16]]

# Aggregate host-side tensor types (all columns concatenated)
params_all_ty = np.ndarray[(N_COLS * PARAMS_SIZE,),  np.dtype[bfloat16]]
spline_all_ty = np.ndarray[(N_COLS * SPLINE_SIZE,),  np.dtype[bfloat16]]
obs_all_ty    = np.ndarray[(N_COLS * OBS_SIZE,),     np.dtype[bfloat16]]
result_all_ty = np.ndarray[(N_COLS * RESULT_SIZE,),  np.dtype[bfloat16]]

KERNEL_SRC = os.path.join(os.path.dirname(__file__),
                           "kernels", "frenet_planner_kernel.cc")


def frenet_planner_design(dev):
    """
    Build and return the MLIR module for the NPU Frenet Planner.

    Parameters
    ----------
    dev : NPU device object (NPU1 or NPU2)

    Returns
    -------
    MLIR module string
    """

    # ── Kernel declaration ────────────────────────────────────────────────────
    frenet_kernel = ExternalFunction(
        "frenet_planner_col",
        source_file=KERNEL_SRC,
        arg_types=[params_ty, spline_ty, obs_ty, result_ty],
        include_dirs=[cxx_header_path()],
    )

    # ── Per-column ObjectFifos ────────────────────────────────────────────────
    # Input FIFOs: shim → compute tile
    of_params  = [ObjectFifo(params_ty, name=f"params{i}")  for i in range(N_COLS)]
    of_spline  = [ObjectFifo(spline_ty, name=f"spline{i}")  for i in range(N_COLS)]
    of_obs     = [ObjectFifo(obs_ty,    name=f"obs{i}")     for i in range(N_COLS)]
    # Output FIFOs: compute tile → shim
    of_result  = [ObjectFifo(result_ty, name=f"result{i}")  for i in range(N_COLS)]

    # ── Worker task (identical logic for all 5 columns) ───────────────────────
    def worker_fn(of_p, of_s, of_o, of_r, kernel):
        for _ in range_(sys.maxsize):
            p = of_p.acquire(1)
            s = of_s.acquire(1)
            o = of_o.acquire(1)
            r = of_r.acquire(1)
            kernel(p, s, o, r)
            of_p.release(1)
            of_s.release(1)
            of_o.release(1)
            of_r.release(1)

    # ── Instantiate one Worker per column ─────────────────────────────────────
    workers = []
    for i in range(N_COLS):
        workers.append(
            Worker(
                worker_fn,
                fn_args=[
                    of_params[i].cons(),
                    of_spline[i].cons(),
                    of_obs[i].cons(),
                    of_result[i].prod(),
                    frenet_kernel,
                ],
            )
        )

    # ── Runtime sequence ──────────────────────────────────────────────────────
    # Host tensors are the concatenation of all per-column buffers so that a
    # single DMA descriptor covers contiguous memory.  The IRON split/join API
    # is not used here because spline and obstacle data is shared (same content
    # broadcast to every worker via separate DMA descriptors).
    rt = Runtime()
    with rt.sequence(
        params_all_ty,   # input 0: all 5 params buffers concatenated
        spline_all_ty,   # input 1: spline  (same content × 5, pre-tiled by host)
        obs_all_ty,      # input 2: obstacles (same content × 5)
        result_all_ty,   # output:  all 5 result buffers concatenated
    ) as (params_in, spline_in, obs_in, result_out):

        rt.start(*workers)

        # Fill per-column params (each slice has different end-states)
        for i in range(N_COLS):
            rt.fill(
                of_params[i].prod(),
                params_in,
                sizes=[1, 1, 1, PARAMS_SIZE],
                offsets=[0, 0, 0, i * PARAMS_SIZE],
            )

        # Fill per-column spline (same data repeated N_COLS times in spline_in)
        for i in range(N_COLS):
            rt.fill(
                of_spline[i].prod(),
                spline_in,
                sizes=[1, 1, 1, SPLINE_SIZE],
                offsets=[0, 0, 0, i * SPLINE_SIZE],
            )

        # Fill per-column obstacles (same data repeated N_COLS times)
        for i in range(N_COLS):
            rt.fill(
                of_obs[i].prod(),
                obs_in,
                sizes=[1, 1, 1, OBS_SIZE],
                offsets=[0, 0, 0, i * OBS_SIZE],
            )

        # Drain per-column results into contiguous output buffer
        for i in range(N_COLS):
            rt.drain(
                of_result[i].cons(),
                result_out,
                sizes=[1, 1, 1, RESULT_SIZE],
                offsets=[0, 0, 0, i * RESULT_SIZE],
                wait=(i == N_COLS - 1),  # sync on last drain only
            )

    return Program(dev, rt).resolve_program(SequentialPlacer())


# ── Stand-alone MLIR generation ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NPU Frenet Planner IRON design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu",
                   help="Target NPU device (default: npu)")
    args = p.parse_args()

    dev = NPU1() if args.dev == "npu" else NPU2()
    print(frenet_planner_design(dev))
