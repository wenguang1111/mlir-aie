//===- frenet_planner_kernel.cc ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// NPU Frenet/FISS+ Planner Kernel – Stages 1-4 combined on one AIE2 tile.
//
// Stage 1 – Frenet polynomial trajectory generation
//   Fits a QuinticPolynomial for lateral motion and a QuarticPolynomial for
//   longitudinal motion, evaluated at N_STEPS time steps.
//
// Stage 2 – Frenet → Cartesian conversion
//   Queries a pre-computed CubicSpline2D reference path (passed in spline_buf)
//   to map (s, d) pairs to world-frame (x, y, cos_yaw, sin_yaw).
//
// Stage 3 – Kinematic constraint check
//   Rejects trajectories where speed > max_speed or |accel| > max_accel.
//
// Stage 4 – Collision detection
//   Uses the Separating Axis Theorem (SAT) between a fixed 4-point rectangular
//   ego-vehicle polygon and 4-point rectangular obstacle polygons.
//   Both polygons share the same vertex count (N_EGO_VERTS = N_OBS_VERTS = 4),
//   matching the scenario simulation data format.
//
// The kernel processes N_TRAJS_PER_COL (25) trajectories and writes the
// lowest-cost collision-free trajectory into result_buf.
//
// ─── Buffer layouts (all bfloat16, stored as float at load/store) ──────────
//
// params_buf  [PARAMS_SIZE = 96 bf16]
//   [0..8]   initial FrenetState: t, s, s_d, s_dd, s_ddd, d, d_d, d_dd, d_ddd
//   [9..83]  end states for 25 trajectories: { end_d, end_sd, end_T } × 25
//   [84]     max_speed (m/s)
//   [85]     max_accel (m/s²)
//   [86]     vehicle_l (m)
//   [87]     vehicle_w (m)
//   [88]     (reserved / padding – was vehicle_margin, no longer used)
//   [89]     target_speed (m/s)
//   [90]     time_step_now   (float encoding of int, used as obstacle time offset)
//   [91]     n_steps         (actual number of time steps to evaluate, ≤ N_STEPS)
//   [92..95] padding
//
// spline_buf  [SPLINE_SIZE = 928 bf16]
//   [0]       n_segs (number of spline segments, ≤ N_SPLINE_SEGS=100)
//   [1..101]  s[0..n_segs]  (arc-length breakpoints, n_segs+1 values)
//   [102..201] ax[0..99]    x-spline a-coefficient per segment
//   [202..301] bx           x-spline b-coefficient
//   [302..401] cx           x-spline c-coefficient
//   [402..501] dx           x-spline d-coefficient
//   [502..601] ay            y-spline a-coefficient
//   [602..701] by
//   [702..801] cy
//   [802..901] dy
//   [902..927] padding
//
// obs_buf  [OBS_SIZE = 3232 bf16]
//   [0]      n_obs      (number of active obstacles, ≤ N_OBS_MAX=8)
//   [1]      n_timesteps (≤ N_STEPS=50)
//   [2 + t*(N_OBS_MAX*N_OBS_VERTS*2) + o*(N_OBS_VERTS*2) + v*2 + c]
//            obstacle vertex data: c=0 → x, c=1 → y
//            Each obstacle is a 4-vertex convex rectangle (N_OBS_VERTS=4).
//
// result_buf  [RESULT_SIZE = 256 bf16]
//   [0]       best_cost  (-1.0 → no valid trajectory found)
//   [1]       best_idx   (0..24 within this column, -1.0 → none)
//   [2..51]   x[0..49]   Cartesian x of best trajectory
//   [52..101]  y[0..49]
//   [102..151] cos_yaw[0..49]
//   [152..201] sin_yaw[0..49]
//   [202..251] s_d[0..49]  longitudinal speed profile
//   [252..255] padding
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <aie_api/aie.hpp>

#include <float.h>
#include <math.h>
#include <stdint.h>

// ============================================================
// Compile-time constants
// ============================================================
#define N_TRAJS_PER_COL  25
#define N_STEPS          50
#define N_SPLINE_SEGS    100
#define N_OBS_MAX        8
#define N_OBS_VERTS      4   // obstacle: axis-aligned rectangle (scenario data)
#define N_EGO_VERTS      4   // ego vehicle: axis-aligned rectangle (same as obstacles)

// Cost weights (matching the WX1 defaults from C_Planner/common/cost)
#define W_T    1.0f
#define W_V    1.0f
#define W_A    0.1f
#define W_J    0.01f
#define W_LC   1.0f

// ============================================================
// Helper: solve 2×2 linear system  A·x = b  (Cramer's rule)
// ============================================================
static inline void solve_2x2(float a00, float a01, float a10, float a11,
                              float b0, float b1,
                              float *x0, float *x1) {
  float det = a00 * a11 - a01 * a10;
  if (det == 0.0f) det = 1e-10f;
  *x0 = (b0 * a11 - b1 * a01) / det;
  *x1 = (a00 * b1 - a10 * b0) / det;
}

// ============================================================
// Helper: solve 3×3 linear system  A·x = b  (Cramer's rule)
// ============================================================
static inline void solve_3x3(float a00, float a01, float a02,
                              float a10, float a11, float a12,
                              float a20, float a21, float a22,
                              float b0, float b1, float b2,
                              float *x0, float *x1, float *x2) {
  float det = a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);
  if (det == 0.0f) det = 1e-10f;
  float inv = 1.0f / det;

  *x0 = inv * ( b0  * (a11 * a22 - a12 * a21)
               - a01 * ( b1 * a22 - a12 *  b2)
               + a02 * ( b1 * a21 - a11 *  b2));

  *x1 = inv * (a00 * ( b1 * a22 - a12 *  b2)
               -  b0 * (a10 * a22 - a12 * a20)
               + a02 * (a10 *  b2 -  b1 * a20));

  *x2 = inv * (a00 * (a11 *  b2 -  b1 * a21)
               - a01 * (a10 *  b2 -  b1 * a20)
               +  b0 * (a10 * a21 - a11 * a20));
}

// ============================================================
// Stage 1a: Quartic polynomial fit for longitudinal motion
//   xs + vxs*t + (axs/2)*t² + a3*t³ + a4*t⁴
//   boundary: vel = vxe, accel = axe at t = T
// ============================================================
static inline void fit_quartic(float xs, float vxs, float axs,
                                float vxe, float axe, float T,
                                float *a0, float *a1, float *a2,
                                float *a3, float *a4) {
  *a0 = xs;
  *a1 = vxs;
  *a2 = 0.5f * axs;

  float T2 = T * T, T3 = T2 * T;
  float b0 = vxe - *a1 - 2.0f * *a2 * T;
  float b1 = axe - 2.0f * *a2;
  solve_2x2(3.0f * T2, 4.0f * T3,
            6.0f * T,  12.0f * T2,
            b0, b1, a3, a4);
}

static inline float eval_quartic(float a0, float a1, float a2, float a3,
                                  float a4, float t) {
  return ((((a4 * t) + a3) * t + a2) * t + a1) * t + a0;
}
static inline float eval_quartic_d1(float a1, float a2, float a3, float a4,
                                     float t) {
  return (((4.0f * a4 * t) + 3.0f * a3) * t + 2.0f * a2) * t + a1;
}
static inline float eval_quartic_d2(float a2, float a3, float a4, float t) {
  return ((12.0f * a4 * t) + 6.0f * a3) * t + 2.0f * a2;
}
static inline float eval_quartic_d3(float a3, float a4, float t) {
  return 24.0f * a4 * t + 6.0f * a3;
}

// ============================================================
// Stage 1b: Quintic polynomial fit for lateral motion
//   xs + vxs*t + (axs/2)*t² + a3*t³ + a4*t⁴ + a5*t⁵
//   boundary: pos = xe, vel = vxe, accel = axe at t = T
// ============================================================
static inline void fit_quintic(float xs, float vxs, float axs,
                                float xe, float vxe, float axe, float T,
                                float *a0, float *a1, float *a2,
                                float *a3, float *a4, float *a5) {
  *a0 = xs;
  *a1 = vxs;
  *a2 = 0.5f * axs;

  float T2 = T * T, T3 = T2 * T, T4 = T2 * T2, T5 = T4 * T;
  float b0 = xe  - *a0 - *a1 * T - *a2 * T2;
  float b1 = vxe - *a1 - 2.0f * *a2 * T;
  float b2 = axe - 2.0f * *a2;
  solve_3x3(       T3,        T4,        T5,
            3.0f * T2, 4.0f * T3, 5.0f * T4,
            6.0f * T, 12.0f * T2, 20.0f * T3,
            b0, b1, b2, a3, a4, a5);
}

static inline float eval_quintic(float a0, float a1, float a2, float a3,
                                  float a4, float a5, float t) {
  return (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t + a0;
}
static inline float eval_quintic_d1(float a1, float a2, float a3, float a4,
                                     float a5, float t) {
  return ((((5.0f * a5 * t) + 4.0f * a4) * t + 3.0f * a3) * t +
          2.0f * a2) * t + a1;
}
static inline float eval_quintic_d2(float a2, float a3, float a4, float a5,
                                     float t) {
  return (((20.0f * a5 * t) + 12.0f * a4) * t + 6.0f * a3) * t +
         2.0f * a2;
}
static inline float eval_quintic_d3(float a3, float a4, float a5, float t) {
  return (60.0f * a5 * t + 24.0f * a4) * t + 6.0f * a3;
}

// ============================================================
// Stage 2: CubicSpline2D lookup
//   Returns x-position, y-position, dx/ds and dy/ds at arc-length s_val.
//   The division by |ds| yields the unit tangent, which we use directly
//   as (cos_yaw, sin_yaw) to avoid calling atan2().
// ============================================================
static inline void spline_query(const float *s_arr, int n_segs,
                                 const float *ax, const float *bx,
                                 const float *cx, const float *dx,
                                 const float *ay, const float *by,
                                 const float *cy, const float *dy,
                                 float s_val,
                                 float *out_x, float *out_y,
                                 float *out_cos_yaw, float *out_sin_yaw) {
  // Clamp s_val to valid range
  if (s_val < s_arr[0])       s_val = s_arr[0];
  if (s_val > s_arr[n_segs])  s_val = s_arr[n_segs];

  // Linear search for segment (sequential access in s → ascending order)
  int i = 0;
  for (int k = 0; k < n_segs - 1; k++) {
    if (s_arr[k + 1] > s_val) { i = k; break; }
    i = k;
  }

  float ds = s_val - s_arr[i];
  float ds2 = ds * ds, ds3 = ds2 * ds;

  *out_x = ax[i] + bx[i] * ds + cx[i] * ds2 + dx[i] * ds3;
  *out_y = ay[i] + by[i] * ds + cy[i] * ds2 + dy[i] * ds3;

  // First derivatives: d(x)/ds and d(y)/ds
  float dxds = bx[i] + 2.0f * cx[i] * ds + 3.0f * dx[i] * ds2;
  float dyds = by[i] + 2.0f * cy[i] * ds + 3.0f * dy[i] * ds2;

  // Normalise to unit tangent → (cos_yaw, sin_yaw)
  float r = sqrtf(dxds * dxds + dyds * dyds);
  if (r < 1e-6f) r = 1e-6f;
  *out_cos_yaw = dxds / r;
  *out_sin_yaw = dyds / r;
}

// ============================================================
// Stage 4: SAT collision check between two convex polygons
//   poly1: n1 vertices (x1[], y1[]),  poly2: n2 vertices (x2[], y2[])
//   Returns 1 if collision detected, 0 if separated.
// ============================================================
static inline int sat_collision(const float *x1, const float *y1, int n1,
                                 const float *x2, const float *y2, int n2) {
  // Test all edge normals from poly1
  for (int i = 0; i < n1; i++) {
    int j = (i + 1) % n1;
    float nx = -(y1[j] - y1[i]);
    float ny =  (x1[j] - x1[i]);

    float mn1 = FLT_MAX, mx1 = -FLT_MAX;
    for (int k = 0; k < n1; k++) {
      float p = x1[k] * nx + y1[k] * ny;
      if (p < mn1) mn1 = p;
      if (p > mx1) mx1 = p;
    }
    float mn2 = FLT_MAX, mx2 = -FLT_MAX;
    for (int k = 0; k < n2; k++) {
      float p = x2[k] * nx + y2[k] * ny;
      if (p < mn2) mn2 = p;
      if (p > mx2) mx2 = p;
    }
    if (mx1 < mn2 || mx2 < mn1) return 0; // Separating axis found
  }

  // Test all edge normals from poly2
  for (int i = 0; i < n2; i++) {
    int j = (i + 1) % n2;
    float nx = -(y2[j] - y2[i]);
    float ny =  (x2[j] - x2[i]);

    float mn1 = FLT_MAX, mx1 = -FLT_MAX;
    for (int k = 0; k < n1; k++) {
      float p = x1[k] * nx + y1[k] * ny;
      if (p < mn1) mn1 = p;
      if (p > mx1) mx1 = p;
    }
    float mn2 = FLT_MAX, mx2 = -FLT_MAX;
    for (int k = 0; k < n2; k++) {
      float p = x2[k] * nx + y2[k] * ny;
      if (p < mn2) mn2 = p;
      if (p > mx2) mx2 = p;
    }
    if (mx1 < mn2 || mx2 < mn1) return 0;
  }
  return 1; // All axes overlap → collision
}

// ============================================================
// Build 4-point rectangular ego-vehicle polygon in world frame.
//
//  Vehicle-local frame (x = forward, y = left):
//
//       P3 ────── P2
//       |          |
//       P0 ────── P1
//
//   P0 = (-L/2, -W/2)   rear-right
//   P1 = (+L/2, -W/2)   front-right
//   P2 = (+L/2, +W/2)   front-left
//   P3 = (-L/2, +W/2)   rear-left
//
// Matches the 4-vertex format used by obstacle polygons in the scenario data.
// After rotating by (cos_yaw, sin_yaw) and translating to (cx, cy):
// ============================================================
static inline void build_ego_polygon(float cx, float cy,
                                      float cos_yaw, float sin_yaw,
                                      float L, float W,
                                      float *px, float *py) {
  // Local corners: {lx, ly}
  const float lx[N_EGO_VERTS] = {-L * 0.5f,  L * 0.5f,  L * 0.5f, -L * 0.5f};
  const float ly[N_EGO_VERTS] = {-W * 0.5f, -W * 0.5f,  W * 0.5f,  W * 0.5f};
  for (int k = 0; k < N_EGO_VERTS; k++) {
    px[k] = cx + lx[k] * cos_yaw - ly[k] * sin_yaw;
    py[k] = cy + lx[k] * sin_yaw + ly[k] * cos_yaw;
  }
}

// ============================================================
// Main kernel entry point
// ============================================================
extern "C" {

void frenet_planner_col(bfloat16 *params_buf,   // [PARAMS_SIZE]
                        bfloat16 *spline_buf,   // [SPLINE_SIZE]
                        bfloat16 *obs_buf,      // [OBS_SIZE]
                        bfloat16 *result_buf) { // [RESULT_SIZE]

  // ── Unpack params ──────────────────────────────────────────
  // Initial Frenet state
  float init_s    = (float)params_buf[1];
  float init_s_d  = (float)params_buf[2];
  float init_s_dd = (float)params_buf[3];
  float init_d    = (float)params_buf[5];
  float init_d_d  = (float)params_buf[6];
  float init_d_dd = (float)params_buf[7];
  // (params_buf[0]=t, [4]=s_ddd, [8]=d_ddd unused in polynomial BCs)

  float max_speed  = (float)params_buf[84];
  float max_accel  = (float)params_buf[85];
  float vehicle_l  = (float)params_buf[86];
  float vehicle_w  = (float)params_buf[87];
  // params_buf[88] reserved
  float target_spd = (float)params_buf[89];
  int   t_now      = (int)(float)params_buf[90];
  int   n_steps    = (int)(float)params_buf[91];
  if (n_steps <= 0 || n_steps > N_STEPS) n_steps = N_STEPS;

  // ── Unpack spline ──────────────────────────────────────────
  int n_segs = (int)(float)spline_buf[0];
  if (n_segs <= 0 || n_segs > N_SPLINE_SEGS) n_segs = N_SPLINE_SEGS;

  // Spline coefficient pointers (read directly from bf16 input; convert to
  // float on access inside the loop via explicit cast)
  // We copy them into static float arrays for faster repeated access.
  static float s_arr[N_SPLINE_SEGS + 1];
  static float ax[N_SPLINE_SEGS], bx[N_SPLINE_SEGS];
  static float cx[N_SPLINE_SEGS], dx_[N_SPLINE_SEGS];
  static float ay[N_SPLINE_SEGS], by[N_SPLINE_SEGS];
  static float cy[N_SPLINE_SEGS], dy_[N_SPLINE_SEGS];

  for (int i = 0; i <= n_segs; i++)
    s_arr[i] = (float)spline_buf[1 + i];
  int base = 1 + N_SPLINE_SEGS + 1; // offset to first coeff array
  for (int i = 0; i < n_segs; i++) {
    ax[i]  = (float)spline_buf[base + 0 * N_SPLINE_SEGS + i];
    bx[i]  = (float)spline_buf[base + 1 * N_SPLINE_SEGS + i];
    cx[i]  = (float)spline_buf[base + 2 * N_SPLINE_SEGS + i];
    dx_[i] = (float)spline_buf[base + 3 * N_SPLINE_SEGS + i];
    ay[i]  = (float)spline_buf[base + 4 * N_SPLINE_SEGS + i];
    by[i]  = (float)spline_buf[base + 5 * N_SPLINE_SEGS + i];
    cy[i]  = (float)spline_buf[base + 6 * N_SPLINE_SEGS + i];
    dy_[i] = (float)spline_buf[base + 7 * N_SPLINE_SEGS + i];
  }

  // ── Unpack obstacle header ─────────────────────────────────
  int n_obs       = (int)(float)obs_buf[0];
  int n_obs_steps = (int)(float)obs_buf[1];
  if (n_obs > N_OBS_MAX)    n_obs = N_OBS_MAX;
  if (n_obs_steps > N_STEPS) n_obs_steps = N_STEPS;

  // ── Per-trajectory working arrays (static → tile local mem) ─
  static float s_arr_t[N_STEPS], s_d_t[N_STEPS];
  static float s_dd_t[N_STEPS], s_ddd_t[N_STEPS];
  static float d_arr_t[N_STEPS], d_d_t[N_STEPS];
  static float d_dd_t[N_STEPS], d_ddd_t[N_STEPS];
  static float x_t[N_STEPS], y_t[N_STEPS];
  static float cos_yaw_t[N_STEPS], sin_yaw_t[N_STEPS];

  // For the best valid trajectory
  static float best_x[N_STEPS], best_y[N_STEPS];
  static float best_cos_yaw[N_STEPS], best_sin_yaw[N_STEPS];
  static float best_s_d[N_STEPS];

  float best_cost = -1.0f;  // negative → no valid trajectory yet
  int   best_idx  = -1;

  // ── Scratch for polygon vertices ───────────────────────────
  float ego_px[N_EGO_VERTS], ego_py[N_EGO_VERTS];
  float obs_px[N_OBS_VERTS], obs_py[N_OBS_VERTS];

  // ══════════════════════════════════════════════════════════
  // Main trajectory loop  (N_TRAJS_PER_COL = 25)
  // ══════════════════════════════════════════════════════════
  for (int ti = 0; ti < N_TRAJS_PER_COL; ti++) {

    // ── Stage 1: Polynomial trajectory generation ──────────
    int param_off = 9 + ti * 3;
    float end_d  = (float)params_buf[param_off + 0];
    float end_sd = (float)params_buf[param_off + 1];
    float T      = (float)params_buf[param_off + 2];
    if (T < 0.1f) T = 0.1f; // guard against degenerate T

    float dt = T / (float)n_steps;

    // Quintic for lateral (d): position/vel/accel BCs at both ends
    float la0, la1, la2, la3, la4, la5;
    fit_quintic(init_d, init_d_d, init_d_dd,
                end_d, 0.0f, 0.0f, T,
                &la0, &la1, &la2, &la3, &la4, &la5);

    // Quartic for longitudinal (s): velocity/accel BCs at both ends
    // (position-free: longitudinal polynomial only constrains speed at T)
    float qa0, qa1, qa2, qa3, qa4;
    fit_quartic(init_s, init_s_d, init_s_dd,
                end_sd, 0.0f, T,
                &qa0, &qa1, &qa2, &qa3, &qa4);

    // Evaluate both polynomials at uniform time steps
    float cost_J_lat  = 0.0f;
    float cost_J_lon  = 0.0f;
    for (int k = 0; k < n_steps; k++) {
      float t = dt * (float)k;
      d_arr_t[k]  = eval_quintic   (la0, la1, la2, la3, la4, la5, t);
      d_d_t[k]    = eval_quintic_d1(la1, la2, la3, la4, la5,      t);
      d_dd_t[k]   = eval_quintic_d2(la2, la3, la4, la5,           t);
      d_ddd_t[k]  = eval_quintic_d3(la3, la4, la5,                t);

      s_arr_t[k]  = eval_quartic   (qa0, qa1, qa2, qa3, qa4,      t);
      s_d_t[k]    = eval_quartic_d1(qa1, qa2, qa3, qa4,           t);
      s_dd_t[k]   = eval_quartic_d2(qa2, qa3, qa4,                t);
      s_ddd_t[k]  = eval_quartic_d3(qa3, qa4,                     t);

      // Accumulate jerk costs
      cost_J_lat += d_ddd_t[k]  * d_ddd_t[k];
      cost_J_lon += s_ddd_t[k] * s_ddd_t[k];
    }
    cost_J_lat *= dt;
    cost_J_lon *= dt;

    // ── Stage 3: Kinematic constraint check ────────────────
    // (done before stage 2 to avoid wasted spline queries)
    int ok = 1;
    for (int k = 0; k < n_steps; k++) {
      if (s_d_t[k] < -0.1f || s_d_t[k] > max_speed) { ok = 0; break; }
      float a = s_dd_t[k];
      if (a > max_accel || a < -max_accel)            { ok = 0; break; }
    }
    if (!ok) continue; // reject – skip stages 2 & 4

    // ── Stage 2: Frenet → Cartesian ────────────────────────
    for (int k = 0; k < n_steps; k++) {
      float ref_x, ref_y, cos_ref, sin_ref;
      spline_query(s_arr, n_segs, ax, bx, cx, dx_, ay, by, cy, dy_,
                   s_arr_t[k], &ref_x, &ref_y, &cos_ref, &sin_ref);

      // Lateral offset perpendicular to tangent:
      //   p = ref + d * perp_tangent
      //   perp_tangent = (-sin_ref, cos_ref)  (left = positive d)
      float d = d_arr_t[k];
      x_t[k] = ref_x - d * sin_ref;
      y_t[k] = ref_y + d * cos_ref;

      // Trajectory heading: finite difference (use reference yaw for k=0)
      // We store cos/sin directly to avoid atan2.
      cos_yaw_t[k] = cos_ref;
      sin_yaw_t[k] = sin_ref;
    }
    // Refine yaw from actual trajectory direction (better heading for SAT)
    for (int k = 0; k < n_steps - 1; k++) {
      float ddx = x_t[k + 1] - x_t[k];
      float ddy = y_t[k + 1] - y_t[k];
      float r   = sqrtf(ddx * ddx + ddy * ddy);
      if (r > 1e-6f) {
        cos_yaw_t[k] = ddx / r;
        sin_yaw_t[k] = ddy / r;
      }
    }

    // ── Stage 4: Collision detection ────────────────────────
    int collision = 0;
    for (int k = 0; k < n_steps && !collision; k++) {
      // Skip odd steps (check_resolution=2 matches C_Planner default)
      if (k % 2 != 0) continue;

      int obs_t = t_now + k;
      if (obs_t >= n_obs_steps) break;

      // Build 4-point ego rectangle at this time step
      build_ego_polygon(x_t[k], y_t[k],
                        cos_yaw_t[k], sin_yaw_t[k],
                        vehicle_l, vehicle_w,
                        ego_px, ego_py);

      // Check against every obstacle at this time step
      for (int o = 0; o < n_obs && !collision; o++) {
        int obs_base = 2 + obs_t * (N_OBS_MAX * N_OBS_VERTS * 2)
                         + o * (N_OBS_VERTS * 2);
        for (int v = 0; v < N_OBS_VERTS; v++) {
          obs_px[v] = (float)obs_buf[obs_base + v * 2 + 0];
          obs_py[v] = (float)obs_buf[obs_base + v * 2 + 1];
        }
        // AABB fast-reject before full SAT
        float omin_x = obs_px[0], omax_x = obs_px[0];
        float omin_y = obs_py[0], omax_y = obs_py[0];
        for (int v = 1; v < N_OBS_VERTS; v++) {
          if (obs_px[v] < omin_x) omin_x = obs_px[v];
          if (obs_px[v] > omax_x) omax_x = obs_px[v];
          if (obs_py[v] < omin_y) omin_y = obs_py[v];
          if (obs_py[v] > omax_y) omax_y = obs_py[v];
        }
        float emin_x = ego_px[0], emax_x = ego_px[0];
        float emin_y = ego_py[0], emax_y = ego_py[0];
        for (int v = 1; v < N_EGO_VERTS; v++) {
          if (ego_px[v] < emin_x) emin_x = ego_px[v];
          if (ego_px[v] > emax_x) emax_x = ego_px[v];
          if (ego_py[v] < emin_y) emin_y = ego_py[v];
          if (ego_py[v] > emax_y) emax_y = ego_py[v];
        }
        // AABB overlap test
        if (emax_x < omin_x || omax_x < emin_x) continue;
        if (emax_y < omin_y || omax_y < emin_y) continue;

        // Full SAT check (AABB passed)
        if (sat_collision(ego_px, ego_py, N_EGO_VERTS,
                          obs_px, obs_py, N_OBS_VERTS))
          collision = 1;
      }
    }
    if (collision) continue; // reject

    // ── Cost function ──────────────────────────────────────
    // Matches WX1 weights from C_Planner/common/cost/cost_function.cpp
    float term_vel = s_d_t[n_steps - 1];
    float cost =  W_J  * (cost_J_lat + cost_J_lon)
               +  W_T  * (10.0f - T)
               +  W_V  * (target_spd - term_vel) * (target_spd - term_vel)
               +  W_LC * (end_d * end_d);

    // ── Update best ────────────────────────────────────────
    if (best_idx == -1 || cost < best_cost) {
      best_cost = cost;
      best_idx  = ti;
      for (int k = 0; k < n_steps; k++) {
        best_x[k]       = x_t[k];
        best_y[k]       = y_t[k];
        best_cos_yaw[k] = cos_yaw_t[k];
        best_sin_yaw[k] = sin_yaw_t[k];
        best_s_d[k]     = s_d_t[k];
      }
    }
  } // end trajectory loop

  // ── Write result buffer ────────────────────────────────────
  result_buf[0] = (bfloat16)best_cost;
  result_buf[1] = (bfloat16)(float)best_idx;
  for (int k = 0; k < N_STEPS; k++) {
    result_buf[2   + k] = (bfloat16)(best_idx >= 0 ? best_x[k]       : 0.0f);
    result_buf[52  + k] = (bfloat16)(best_idx >= 0 ? best_y[k]       : 0.0f);
    result_buf[102 + k] = (bfloat16)(best_idx >= 0 ? best_cos_yaw[k] : 0.0f);
    result_buf[152 + k] = (bfloat16)(best_idx >= 0 ? best_sin_yaw[k] : 0.0f);
    result_buf[202 + k] = (bfloat16)(best_idx >= 0 ? best_s_d[k]     : 0.0f);
  }
}

} // extern "C"
