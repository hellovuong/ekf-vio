# EKF-VIO — Complete Mathematical Reference

---

## 0. System Overview

The filter fuses two sensors running at different rates:

| Sensor | Rate | Role |
|--------|------|------|
| IMU (gyro + accel) | ~200 Hz | **Predict** — propagates pose, velocity, biases |
| Stereo camera | ~20–30 Hz | **Update** — corrects state via reprojection residuals |

The architecture is a **tightly-coupled Error-State EKF** (ES-EKF):
- The *nominal state* is propagated with full nonlinear kinematics (RK4).
- The *error state* (small perturbation δx) is kept Gaussian and propagated linearly.
- The EKF covariance P lives entirely in error-state space (15×15).

---

## 1. State Definition

### Nominal state (what we track)

| Symbol | Size | Description | Frame |
|--------|------|-------------|-------|
| `p` | 3 | Position | World |
| `v` | 3 | Velocity | World |
| `R_wb` (via `T_wb`) | SO(3) | Rotation world←body, Hamilton convention | — |
| `b_g` | 3 | Gyroscope bias | Body |
| `b_a` | 3 | Accelerometer bias | Body |

The full pose is stored as `Sophus::SE3d T_wb` (world←body).

### Error state (what the EKF linearises over)

```
δx = [δp, δv, δθ, δb_g, δb_a]  ∈ ℝ¹⁵
      0:3  3:6  6:9  9:12  12:15
```

`δθ ∈ ℝ³` is the rotation error vector (axis-angle), related to the true rotation by:

```
R_true = R_nominal · Exp(δθ)
```

where `Exp(·) : ℝ³ → SO(3)` is the matrix exponential on SO(3) (`Sophus::SO3d::exp`).

### Covariance

```
P ∈ ℝ¹⁵ˣ¹⁵   (symmetric positive-definite)
```

Indexed in the same order as δx above.

---

## 2. IMU Measurement Model

Raw IMU readings contain sensor noise and slowly-varying biases:

```
ω_meas = ω_true + b_g + n_g       (gyroscope)
a_meas = a_true + b_a + n_a       (accelerometer)
```

**Bias-corrected inputs** used for integration:

```
ω_c = ω_meas − b_g     (corrected angular velocity)
a_c = a_meas − b_a     (corrected specific force)
```

**Noise spectral densities** (continuous time, from Allan variance / datasheet):

| Parameter | Symbol | Typical (EuRoC) | Units |
|-----------|--------|-----------------|-------|
| Gyro noise | σ_g | 1.7×10⁻⁴ | rad/s/√Hz |
| Accel noise | σ_a | 2.0×10⁻³ | m/s²/√Hz |
| Gyro bias walk | σ_bg | 1.9×10⁻⁵ | rad/s²/√Hz |
| Accel bias walk | σ_ba | 3.0×10⁻³ | m/s³/√Hz |

---

## 3. Continuous-Time Kinematics (Nominal State)

The IMU mechanisation equations in the world frame:

```
ṗ = v
v̇ = R_wb · a_c + g          (specific force rotated to world + gravity)
Ṙ = R_wb · [ω_c]×           (rotation ODE on SO(3))
ḃ_g = 0                      (random walk modelled as noise only)
ḃ_a = 0
```

where `g = [0, 0, −9.81]ᵀ` m/s² (world frame, z-up convention).

`[ω]×` denotes the 3×3 skew-symmetric matrix of ω:

```
[ω]× = [ 0    -ω_z   ω_y ]
       [ ω_z   0    -ω_x ]
       [-ω_y   ω_x   0   ]
```

---

## 4. Predict — Full RK4 Integration

The predict step integrates three coupled ODEs simultaneously over each IMU interval dt:
the **nominal state** (p, v, R), the **transition matrix** Φ, and the **discrete noise** Q_d.

### 4.1 IMU Bracketing and Midpoint Interpolation

Two IMU readings bracket the interval: `[imu_prev, imu_curr]`.
Linear interpolation gives the midpoint:

```
ω_start = ω_prev − b_g,    a_start = a_prev − b_a
ω_end   = ω_curr − b_g,    a_end   = a_curr − b_a
ω_mid   = ½·(ω_start + ω_end)
a_mid   = ½·(a_start + a_end)
```

Exact intermediate rotations are pre-computed via SO(3) exponential (no linearisation):

```
R_mid = R₀ · Exp(ω_mid · ½·dt)
R_end = R₀ · Exp(ω_end · dt)
```

### 4.2 RK4 Stage Evaluation

Each stage computes derivatives `{dp, dv, dΦ, dQ_d}` given a (state, Φ, Q_d, IMU) tuple.
The per-stage derivative function `evalDeriv` evaluates:

```
dp      = v
dv      = R · a_c + g
F       = computeF(ω_c, a_c, R)    ← F re-evaluated at each stage's rotation
dΦ      = F · Φ                    ← transition matrix ODE:  Φ̇ = F·Φ
dQ_d    = F·Q_d + Q_d·Fᵀ + GQGᵀ  ← Lyapunov-like noise ODE
```

The four stages with their bracketed IMU readings and initial conditions `Φ(0) = I`, `Q_d(0) = 0`:

```
k1: evalDeriv(ω_start, a_start, pvq₀, I,              0           )
k2: evalDeriv(ω_mid,   a_mid,   pvq₂, I + ½dt·k1.dΦ, ½dt·k1.dQ_d)   pvq₂.R = R_mid
k3: evalDeriv(ω_mid,   a_mid,   pvq₃, I + ½dt·k2.dΦ, ½dt·k2.dQ_d)   pvq₃.R = R_mid
k4: evalDeriv(ω_end,   a_end,   pvq₄, I +  dt·k3.dΦ,  dt·k3.dQ_d)   pvq₄.R = R_end
```

### 4.3 Weighted Sum and State/Covariance Update

RK4 weighted combination `(1/6)·(k1 + 2·k2 + 2·k3 + k4)`:

```
p_new = p₀ + (dt/6)·(dp₁ + 2·dp₂ + 2·dp₃ + dp₄)
v_new = v₀ + (dt/6)·(dv₁ + 2·dv₂ + 2·dv₃ + dv₄)
R_new = R_end                          ← exact SO(3) rotation, no approximation

Φ   = I   + (dt/6)·(dΦ₁  + 2·dΦ₂  + 2·dΦ₃  + dΦ₄ )    O(dt⁵) error
Q_d =       (dt/6)·(dQ₁  + 2·dQ₂  + 2·dQ₃  + dQ₄ )    O(dt⁵) error

P_new = Φ · P_old · Φᵀ + Q_d
P_new = ½·(P_new + P_newᵀ)            ← force symmetry
```

---

## 5. Continuous-Time Error-State Jacobian F (15×15)

Linearising the kinematics around the nominal trajectory gives the error-state ODE:

```
δẋ = F(x) · δx + G · n
```

where `n = [n_a, n_g, n_bg, n_ba]ᵀ` is the stacked noise vector.

**F matrix** (only non-zero 3×3 blocks shown):

```
       δp     δv     δθ           δb_g    δb_a
δṗ  [  0      I      0             0       0   ]   row 0:3
δv̇  [  0      0   −R_wb·[a_c]×    0    −R_wb  ]   row 3:6
δθ̇  [  0      0   −[ω_c]×        −I      0   ]   row 6:9
δḃ_g[  0      0      0             0       0   ]   row 9:12
δḃ_a[  0      0      0             0       0   ]   row 12:15
```

Key blocks:

| Block | Expression | Physics |
|-------|-----------|---------|
| F[0:3, 3:6] | I | Velocity drives position |
| F[3:6, 6:9] | −R_wb·[a_c]× | Rotation error couples into velocity via skew of specific force |
| F[3:6, 12:15] | −R_wb | Accel bias error drives velocity error |
| F[6:9, 6:9] | −[ω_c]× | Gyro reading drives rotation error (attitude dynamics) |
| F[6:9, 9:12] | −I | Gyro bias error drives rotation error |

**Important:** F is re-evaluated at each RK4 stage using the stage's intermediate rotation R,
not the current state rotation. This is what makes the full RK4 accurate — F(x) changes as
R evolves through the interval.

---

## 6. Continuous Noise Input Matrix G and GQGᵀ

The noise input matrix G maps raw sensor noises into error-state space:

```
G =  [ 0    0    0    0  ]   rows 0:3   (position — not directly excited)
     [-R    0    0    0  ]   rows 3:6   (velocity ← accel noise)
     [ 0   -I    0    0  ]   rows 6:9   (rotation ← gyro noise)
     [ 0    0    I    0  ]   rows 9:12  (gyro bias ← bias walk)
     [ 0    0    0    I  ]   rows 12:15 (accel bias ← bias walk)
```

**Continuous process noise covariance Q_c:**

```
Q_c = diag(σ_a²·I₃, σ_g²·I₃, σ_bg²·I₃, σ_ba²·I₃)    (12×12)
```

**Key insight — GQ_cGᵀ is rotation-invariant:**

Every G block touching rotation is either `±R` or `±I`.
Since `R · σ²I · Rᵀ = σ²·R·Rᵀ = σ²·I`, rotation cancels out.
Therefore **GQ_cGᵀ is a constant matrix** — computed once in the constructor:

```
GQGᵀ = diag(0,  σ_a²·I₃,  σ_g²·I₃,  σ_bg²·I₃,  σ_ba²·I₃)    (15×15)
         p     v           θ          b_g          b_a
```

This saves recomputing a 15×15 matrix product at every 200 Hz predict step.

---

## 7. Coordinate Frames and Transforms

```
World (w) ←──── T_wb ────── Body/IMU (b) ←──── T_cam_imu ────── Camera (c)
```

| Transform | Direction | Storage |
|-----------|-----------|---------|
| `T_wb` | world ← body | `state_.T_wb` (SE3d) |
| `T_cam_imu` | camera ← IMU/body | `cam_.T_cam_imu` (SE3d, fixed extrinsic) |

**World → Camera:**
```
p_c = T_cam_imu · T_wb⁻¹ · p_w
    = R_ci · R_wbᵀ · (p_w − t_wb) + t_ci
```

**Camera → World:**
```
p_w = T_wb · T_cam_imu⁻¹ · p_c
```

---

## 8. Stereo Projection Model

For a 3D point `p_c = [x, y, z]ᵀ` in the left camera frame:

```
Left image:   u_l = fx·x/z + cx,      v_l = fy·y/z + cy
Right image:  u_r = fx·(x−b)/z + cx,  v_r = v_l      (rectified horizontal stereo)
```

where `b` is the stereo baseline (metres). The right camera shifts the x-coordinate only —
`v_r = v_l` because the cameras are rectified to the same horizontal plane.

**Stereo triangulation** (inverse of projection):

```
d = u_l − u_r          (disparity, always > 0 for points in front)
Z = fx·b / d
X = (u_l − cx)·Z / fx
Y = (v_l − cy)·Z / fy
```

---

## 9. Measurement Update — Stereo Reprojection

### 9.1 Observation model

Each tracked landmark with world position `p_w` produces a 4D measurement:

```
z_i = [u_l, v_l, u_r, v_r]ᵀ              (observed stereo pixels)
ẑ_i = π(T_cam_imu · T_wb⁻¹ · p_w)        (predicted stereo pixels)

r_i = z_i − ẑ_i ∈ ℝ⁴                    (innovation / residual)
```

### 9.2 Measurement Jacobian H_i (4×15)

By the chain rule: `∂z/∂δx = (∂π/∂p_c) · (∂p_c/∂δx)`

**Projection Jacobians** `∂π/∂p_c` (2×3 each):

```
J_l = [ fx/z    0    −fx·x/z² ]    (left camera)
      [  0     fy/z  −fy·y/z² ]

J_r = [ fx/z    0    −fx·(x−b)/z² ]   (right camera; only x-column differs)
      [  0     fy/z  −fy·y/z²     ]
```

**Pose-to-point Jacobians** `∂p_c/∂δx`:

```
∂p_c/∂δp  = −R_cw = −R_ci · R_wbᵀ        (3×3)

∂p_c/∂δθ  = R_ci · [p_imu]×              (3×3)
```

where `p_imu = R_wbᵀ·(p_w − t_wb)` is the landmark in body/IMU frame.

*Derivation of `∂p_c/∂δθ`:*
Perturb `R_wb → R_wb·Exp(δθ)`:
```
p_c = R_ci·(R_wb·Exp(δθ))ᵀ·(p_w − t_wb)
    ≈ R_ci·(I − [δθ]×)·R_wbᵀ·(p_w − t_wb)
    = R_ci·p_imu − R_ci·[δθ]×·p_imu
    = R_ci·p_imu + R_ci·[p_imu]×·δθ        (using [a]×b = −[b]×a)
```

Therefore `∂p_c/∂δθ = R_ci·[p_imu]×`.

**Full H_i (4×15)** — non-zero only at position (cols 0:3) and orientation (cols 6:9):

```
H_i = [ J_l·(∂p_c/∂δp)   0   J_l·(∂p_c/∂δθ)   0   0 ]   ← left  rows 0:2
      [ J_r·(∂p_c/∂δp)   0   J_r·(∂p_c/∂δθ)   0   0 ]   ← right rows 2:4
```

### 9.3 Outlier Gating (three stages)

Applied before the Kalman update — reject bad measurements cheaply:

**Gate 1 — Depth check:**
```
0.1 m < z_c < 50 m
```
Discards points behind the camera or numerically degenerate.

**Gate 2 — Pixel magnitude:**
```
‖r_i‖_∞ < 40 px
```
Fast component-wise check. Catches gross outliers regardless of P size.
When P is large, S = HPHᵀ + R is also large, so even 100-px residuals can pass the
chi-squared test. This hard gate catches them unconditionally first.

**Gate 3 — Mahalanobis distance:**
```
S_i  = H_i · P · H_iᵀ + R_i      (4×4 innovation covariance)
d²   = r_iᵀ · S_i⁻¹ · r_i        (chi-squared distributed with 4 DOF)
Accept if d² < χ²(4, 0.99) = 9.488
```

### 9.4 Sequential Kalman Update

Measurements are fused **one at a time**, not batched. For N = 200 features:

| Approach | Matrix size | Cost |
|----------|-------------|------|
| Batch | 4N × 4N inversion | O((4N)³) ≈ 512M FLOP |
| Sequential | 4×4 Cholesky × N | O(N·15²) ≈ 2M FLOP |

**Critical:** state and P must be updated after **each** measurement, not accumulated.
If corrections are deferred, P shrinks after the first step but later residuals were
computed at the prior — early features dominate and later measurements are ignored.

**Per-measurement step:**

```
S_k  = H_k · P · H_kᵀ + R              (4×4 innovation covariance)
K_k  = P · H_kᵀ · S_k⁻¹               (15×4 Kalman gain)
dx_k = K_k · r_k                        (15×1 state correction)
```

**State update on the manifold:**
```
p    ← p    + dx[0:3]
v    ← v    + dx[3:6]
R_wb ← R_wb · Exp(dx[6:9])    ← right-multiply: error is in body frame
b_g  ← b_g  + dx[9:12]
b_a  ← b_a  + dx[12:15]
```

**Covariance update — Joseph form:**
```
IKH = I − K_k · H_k                                 (15×15)
P   = IKH · P · IKHᵀ + K_k · R · K_kᵀ             (Joseph form)
P   = ½·(P + Pᵀ)                                    (force symmetry)
```

The Joseph form is algebraically equivalent to the standard `P ← (I−KH)P` but
**guarantees P stays PSD** even with finite-precision arithmetic — the extra `KRKᵀ`
term compensates for round-off in `(I−KH)`.

### 9.5 Measurement Noise

```
R_i = σ_pixel² · I₄    (isotropic, same noise for all 4 pixel observations)
```

Typical value: `σ_pixel = 1.5 px`.

---

## 10. Landmark Management

**First observation:** initialise world position from stereo triangulation using
the current state. Skip EKF update this frame — need a second observation to form
a residual with respect to a *prior* position.

**Subsequent observations:** landmark exists in map → compute residual → update.

**Landmark refresh:** after each update, reproject the triangulated `p_c` back to
world using the *updated* (posterior) state. This keeps the map consistent with
the corrected estimate.

**Culling:** landmarks not observed for more than `landmark_max_age` frames are
removed. This bounds map size and prevents stale geometric constraints.

---

## 11. Loosely-Coupled Pose Update (`updateFromPose`)

An external 6-DOF pose estimate (e.g. from visual odometry) can be fused as an
additional measurement. The observation model is trivial:

```
z = [p_meas − p,  Log(R_wbᵀ · R_meas)]     (6×1 residual)

H = [ I₃   0   0   0   0 ]   ← position rows 0:3
    [ 0    0   I₃  0   0 ]   ← orientation rows 3:6
```

Chi-squared gate: `χ²(6 DOF, 95%) = 12.59`

Standard Kalman update with LDLT decomposition on the 6×6 innovation matrix.

---

## 12. Numerical Safeguards

| Safeguard | Where | Why |
|-----------|-------|-----|
| `P = ½(P + Pᵀ)` | After every covariance update | Floating-point asymmetry accumulates over thousands of steps |
| Joseph form | Covariance update | Guarantees PSD even with imperfect `I−KH` |
| Cholesky on S (not LU) | Kalman gain solve | S is PD by construction; Cholesky is faster and more stable |
| `P.allFinite()` check | After update | Detects NaN/Inf propagation; resets to diagonal |
| Pixel gate before Mahalanobis | Outlier rejection | Large P makes χ² test permissive; raw pixel check is unconditional |
| `dx.allFinite()` | State correction | Skip NaN correction rather than corrupt state |

---

*Generated from source: `src/ekf_rk4.cpp`, `include/ekf_vio/types.hpp`*
