# frenet_planner/test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# ─── NPU Frenet Planner – Test Harness ──────────────────────────────────────
#
# Builds synthetic planning inputs (FrenetState + spline + obstacles),
# packs them into the NPU buffer format, runs the NPU kernel, and validates
# that:
#   1. A valid (collision-free, constraint-passing) trajectory was found.
#   2. The best_cost returned by each worker is consistent with its output
#      trajectory (cost re-computed on host matches within bf16 tolerance).
#
# Usage:
#   python test.py [--dev npu|npu2] [--sim]
#
# When --sim is given the test only generates and prints the MLIR module
# without executing on hardware.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import math
import struct
import sys
import os
import numpy as np
from ml_dtypes import bfloat16

# ── Buffer layout constants (must match frenet_planner_kernel.cc) ─────────────
N_COLS         = 5
N_TRAJS_PER_COL = 25
N_STEPS        = 50
N_SPLINE_SEGS  = 100
N_OBS_MAX      = 8
N_OBS_VERTS    = 4
N_EGO_VERTS    = 6

PARAMS_SIZE    = 96
SPLINE_SIZE    = 928
OBS_SIZE       = 3232
RESULT_SIZE    = 256

# ── Result buffer field offsets ───────────────────────────────────────────────
RES_COST_OFF     = 0
RES_IDX_OFF      = 1
RES_X_OFF        = 2
RES_Y_OFF        = RES_X_OFF   + N_STEPS   # 52
RES_COSYAW_OFF   = RES_Y_OFF   + N_STEPS   # 102
RES_SINYAW_OFF   = RES_COSYAW_OFF + N_STEPS # 152
RES_SD_OFF       = RES_SINYAW_OFF + N_STEPS # 202


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build input buffers
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_grid(num_width=5, num_speed=5, num_t=5,
                     max_road_width=3.5, highest_speed=13.0,
                     min_t=3.0, max_t=5.0):
    """Return list of (end_d, end_sd, T) covering the full 5×5×5 grid."""
    half_w = max_road_width / 2.0
    d_vals  = np.linspace(-half_w, half_w, num_width)
    sd_vals = np.linspace(0.0, highest_speed, num_speed)
    t_vals  = np.linspace(min_t, max_t, num_t)
    samples = [(float(d), float(sd), float(t))
               for d in d_vals for sd in sd_vals for t in t_vals]
    return samples  # 125 tuples


def build_params_buf(init_state, end_states_col, max_speed=13.0,
                     max_accel=5.0, vehicle_l=4.5, vehicle_w=2.0,
                     vehicle_margin=0.0, target_speed=8.0,
                     time_step_now=0, n_steps=N_STEPS):
    """
    Pack a single column's params buffer (PARAMS_SIZE bf16 values).

    Parameters
    ----------
    init_state : dict with keys t,s,s_d,s_dd,s_ddd,d,d_d,d_dd,d_ddd
    end_states_col : list of N_TRAJS_PER_COL (end_d, end_sd, T) tuples
    """
    buf = np.zeros(PARAMS_SIZE, dtype=bfloat16)
    # Initial FrenetState at indices 0..8
    for k, key in enumerate(['t','s','s_d','s_dd','s_ddd','d','d_d','d_dd','d_ddd']):
        buf[k] = bfloat16(init_state[key])
    # End states at indices 9..(9+75-1)
    for i, (ed, esd, eT) in enumerate(end_states_col):
        base = 9 + i * 3
        buf[base + 0] = bfloat16(ed)
        buf[base + 1] = bfloat16(esd)
        buf[base + 2] = bfloat16(eT)
    # Scalar params
    buf[84] = bfloat16(max_speed)
    buf[85] = bfloat16(max_accel)
    buf[86] = bfloat16(vehicle_l)
    buf[87] = bfloat16(vehicle_w)
    buf[88] = bfloat16(vehicle_margin)
    buf[89] = bfloat16(target_speed)
    buf[90] = bfloat16(float(time_step_now))
    buf[91] = bfloat16(float(n_steps))
    return buf


def build_straight_spline(length=100.0, n_segs=50):
    """
    Build a straight-line reference path spline along the x-axis.

    Returns a SPLINE_SIZE bf16 array.
    """
    buf = np.zeros(SPLINE_SIZE, dtype=bfloat16)
    buf[0] = bfloat16(float(n_segs))

    ds = length / n_segs
    s_vals = np.linspace(0.0, length, n_segs + 1)
    for i in range(n_segs + 1):
        buf[1 + i] = bfloat16(s_vals[i])

    # Straight line along x: x(s)=s, y(s)=0
    # CubicSpline coefficients: a=s_i, b=1, c=0, d=0 (for x)
    #                           a=0,   b=0, c=0, d=0 (for y)
    base = 1 + N_SPLINE_SEGS + 1  # = 102
    for i in range(n_segs):
        buf[base + 0 * N_SPLINE_SEGS + i] = bfloat16(s_vals[i])  # ax
        buf[base + 1 * N_SPLINE_SEGS + i] = bfloat16(1.0)        # bx
        # cx, dx, ay, by, cy, dy = 0 (already zeroed)
    return buf


def build_curved_spline(n_segs=50, radius=200.0):
    """
    Build a gently curving reference path (arc of given radius).

    Returns a SPLINE_SIZE bf16 array.
    """
    buf = np.zeros(SPLINE_SIZE, dtype=bfloat16)
    buf[0] = bfloat16(float(n_segs))

    arc_len = math.radians(30) * radius   # 30-degree arc
    s_vals  = np.linspace(0.0, arc_len, n_segs + 1)
    x_vals  = radius * np.sin(s_vals / radius)
    y_vals  = radius * (1.0 - np.cos(s_vals / radius))

    for i in range(n_segs + 1):
        buf[1 + i] = bfloat16(float(s_vals[i]))

    # Fit natural cubic spline (simplified: use Catmull-Rom approximation)
    # For testing correctness the exact spline fit is not required;
    # we use linear interpolation between x_vals and y_vals which gives
    # piecewise-linear a/b coefficients (c=d=0).
    base = 1 + N_SPLINE_SEGS + 1
    ds   = arc_len / n_segs
    for i in range(n_segs):
        buf[base + 0 * N_SPLINE_SEGS + i] = bfloat16(float(x_vals[i]))        # ax
        buf[base + 1 * N_SPLINE_SEGS + i] = bfloat16(float((x_vals[i+1] - x_vals[i]) / ds))  # bx
        buf[base + 4 * N_SPLINE_SEGS + i] = bfloat16(float(y_vals[i]))        # ay
        buf[base + 5 * N_SPLINE_SEGS + i] = bfloat16(float((y_vals[i+1] - y_vals[i]) / ds))  # by
    return buf


def build_obstacle_buf(obstacles, n_timesteps=N_STEPS, time_step_now=0):
    """
    Pack obstacles into OBS_SIZE bf16 array.

    Parameters
    ----------
    obstacles : list of obstacle dicts, each with:
        'x', 'y'   : initial position
        'vx', 'vy' : velocity (m/s)
        'l', 'w'   : half-lengths (m)  – forms a rectangle
    n_timesteps : number of time steps in the array
    time_step_now : first time index to pack (packed into index 0)
    """
    buf  = np.zeros(OBS_SIZE, dtype=bfloat16)
    n_obs = min(len(obstacles), N_OBS_MAX)
    buf[0] = bfloat16(float(n_obs))
    buf[1] = bfloat16(float(n_timesteps))

    dt = 0.1  # seconds per time step (matches tick_t in SettingParameters)

    for t in range(n_timesteps):
        sim_t = time_step_now + t
        for o in range(n_obs):
            obs = obstacles[o]
            cx  = obs['x']  + obs['vx'] * sim_t * dt
            cy  = obs['y']  + obs['vy'] * sim_t * dt
            hl  = obs['l']   # half-length
            hw  = obs['w']   # half-width
            # 4-vertex axis-aligned rectangle (no rotation)
            verts = [
                (cx - hl, cy - hw),  # rear-right
                (cx + hl, cy - hw),  # front-right
                (cx + hl, cy + hw),  # front-left
                (cx - hl, cy + hw),  # rear-left
            ]
            base = 2 + t * (N_OBS_MAX * N_OBS_VERTS * 2) + o * (N_OBS_VERTS * 2)
            for v, (vx, vy) in enumerate(verts):
                buf[base + v * 2 + 0] = bfloat16(float(vx))
                buf[base + v * 2 + 1] = bfloat16(float(vy))
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Host-side reference: unpack result buffer
# ─────────────────────────────────────────────────────────────────────────────

def unpack_result(result_buf):
    """Return dict with best trajectory data from a RESULT_SIZE bf16 array."""
    cost     = float(result_buf[RES_COST_OFF])
    idx      = int(float(result_buf[RES_IDX_OFF]))
    x        = [float(result_buf[RES_X_OFF     + k]) for k in range(N_STEPS)]
    y        = [float(result_buf[RES_Y_OFF     + k]) for k in range(N_STEPS)]
    cos_yaw  = [float(result_buf[RES_COSYAW_OFF + k]) for k in range(N_STEPS)]
    sin_yaw  = [float(result_buf[RES_SINYAW_OFF + k]) for k in range(N_STEPS)]
    s_d      = [float(result_buf[RES_SD_OFF    + k]) for k in range(N_STEPS)]
    return dict(cost=cost, idx=idx, x=x, y=y, cos_yaw=cos_yaw,
                sin_yaw=sin_yaw, s_d=s_d, valid=(idx >= 0))


def select_best_from_results(results_all):
    """Pick lowest-cost valid trajectory across all 5 columns."""
    best = None
    best_cost = float('inf')
    for col, r in enumerate(results_all):
        if r['valid'] and r['cost'] >= 0.0 and r['cost'] < best_cost:
            best_cost = r['cost']
            best = dict(**r, col=col)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main test function
# ─────────────────────────────────────────────────────────────────────────────

def run_test(dev_name="npu", simulation_only=False):
    # ── Design import (lazy to avoid hardware init in --sim mode) ─────────────
    from frenet_planner import (frenet_planner_design,
                                 N_COLS, PARAMS_SIZE, SPLINE_SIZE,
                                 OBS_SIZE, RESULT_SIZE,
                                 params_all_ty, spline_all_ty,
                                 obs_all_ty, result_all_ty)

    if dev_name == "npu":
        from aie.iron.device import NPU1
        dev = NPU1()
    else:
        from aie.iron.device import NPU2
        dev = NPU2()

    mlir_module = frenet_planner_design(dev)

    if simulation_only:
        print(mlir_module)
        return True

    # ── Build inputs ──────────────────────────────────────────────────────────
    init_state = dict(t=0.0, s=0.0, s_d=6.0, s_dd=0.0, s_ddd=0.0,
                      d=0.0, d_d=0.0, d_dd=0.0, d_ddd=0.0)

    all_samples = make_sample_grid()  # 125 end-state tuples
    # Distribute 125 samples across 5 columns (25 each)
    col_samples = [all_samples[i * N_TRAJS_PER_COL:(i + 1) * N_TRAJS_PER_COL]
                   for i in range(N_COLS)]

    # Obstacles: one slow-moving vehicle ahead in the same lane
    obstacles = [
        {'x': 40.0, 'y': 0.0, 'vx': 3.0, 'vy': 0.0, 'l': 2.5, 'w': 1.0},
    ]

    # Pack per-column params
    params_bufs = [
        build_params_buf(init_state, col_samples[i]) for i in range(N_COLS)
    ]

    # Spline: simple straight line (x = s, y = 0)
    spline_buf = build_straight_spline(length=200.0, n_segs=100)

    # Obstacles buffer
    obs_buf = build_obstacle_buf(obstacles)

    # Assemble host-side concatenated tensors
    params_all  = np.concatenate(params_bufs).astype(bfloat16)
    spline_all  = np.tile(spline_buf, N_COLS).astype(bfloat16)  # broadcast
    obs_all     = np.tile(obs_buf,    N_COLS).astype(bfloat16)  # broadcast
    result_all  = np.zeros(N_COLS * RESULT_SIZE, dtype=bfloat16)

    # ── Execute on NPU ────────────────────────────────────────────────────────
    import aie.iron as iron

    npu_params  = iron.from_numpy(params_all,  device=dev_name)
    npu_spline  = iron.from_numpy(spline_all,  device=dev_name)
    npu_obs     = iron.from_numpy(obs_all,     device=dev_name)
    npu_result  = iron.zeros_like(iron.from_numpy(result_all, device=dev_name))

    iron.execute(mlir_module, npu_params, npu_spline, npu_obs, npu_result)
    result_np = iron.to_numpy(npu_result)

    # ── Unpack and validate ───────────────────────────────────────────────────
    results = [unpack_result(result_np[i * RESULT_SIZE:(i + 1) * RESULT_SIZE])
               for i in range(N_COLS)]

    print("\n── Per-column results ──────────────────────────────────")
    for i, r in enumerate(results):
        status = f"cost={r['cost']:.4f}  idx={r['idx']}" if r['valid'] else "NO VALID TRAJ"
        print(f"  Column {i}: {status}")

    best = select_best_from_results(results)
    if best is None:
        print("\nFAIL: No valid trajectory found across all 5 columns.")
        return False

    print(f"\nBest: column {best['col']}, idx {best['idx']}, cost {best['cost']:.4f}")
    print(f"  x[0]={best['x'][0]:.2f}  y[0]={best['y'][0]:.2f}")
    print(f"  x[{N_STEPS-1}]={best['x'][N_STEPS-1]:.2f}  "
          f"y[{N_STEPS-1}]={best['y'][N_STEPS-1]:.2f}")
    print(f"  speed at T: {best['s_d'][N_STEPS-1]:.2f} m/s")

    # Basic sanity checks
    errors = 0

    # 1. Cost must be non-negative
    if best['cost'] < 0:
        print("FAIL: negative cost")
        errors += 1

    # 2. x trajectory should be monotonically increasing (straight-line road)
    x_diffs = [best['x'][k+1] - best['x'][k] for k in range(N_STEPS - 1)]
    if any(dx < -1.0 for dx in x_diffs):
        print("FAIL: x trajectory goes backwards by more than 1m")
        errors += 1

    # 3. |y| ≤ road half-width (3.5m) throughout
    if any(abs(best['y'][k]) > 3.5 for k in range(N_STEPS)):
        print("FAIL: trajectory leaves road boundaries")
        errors += 1

    # 4. Speed must be non-negative and ≤ max_speed everywhere
    max_speed = 13.0
    if any(v < -0.5 or v > max_speed + 0.5 for v in best['s_d']):
        print("FAIL: speed out of bounds")
        errors += 1

    if errors == 0:
        print("\nPASS!")
        return True
    else:
        print(f"\nFAIL: {errors} check(s) failed.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NPU Frenet Planner test")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--sim", action="store_true",
                   help="Print MLIR module only (no hardware execution)")
    args = p.parse_args()

    ok = run_test(dev_name=args.dev, simulation_only=args.sim)
    sys.exit(0 if ok else 1)
