"""Module 5 - Stability and control analysis of the CYCLOPS quadrotor.

1. Static / dynamic stability of the bare airframe: linearise the 6-DOF plant at
   hover and inspect the eigenvalues (no controller).
2. Linear frequency-domain design check of the cascaded attitude loop
   (motor/ESC lag -> inertia -> rate PID -> attitude P): gain and phase margins.
3. Closed-loop step responses on the full NON-linear simulator:
      roll  (lateral / roll stability and control)
      pitch (longitudinal control)
      yaw   (directional stability and control)
      altitude, and gust (disturbance) rejection.
Outputs: docs/figures/stability_*.png and results/stability.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from cyclops.config import Config, G  # noqa: E402
from cyclops.control.flight_controller import FlightController  # noqa: E402
from cyclops.geometry import quat_from_euler, rot_to_euler  # noqa: E402
from cyclops.vehicle.quadrotor import Quadrotor  # noqa: E402

FIG = os.path.join(ROOT, 'docs', 'figures')
RES = os.path.join(ROOT, 'results')


# ------------------------------------------------------------------ 1. open loop
def plant_derivative(cfg, x, w):
    """Continuous-time rigid body dynamics f(x, u). x = [p, v, euler, omega], u = rotor speeds."""
    af = cfg.airframe
    quad = Quadrotor(af, cfg.battery)
    p, v, e, om = x[:3], x[3:6], x[6:9], x[9:12]
    quad.p, quad.v = p.copy(), v.copy()
    quad.q = quat_from_euler(*e)
    quad.w = om.copy()
    T = af.kf * w ** 2
    thrust, torque = quad.mixer.wrench(T)
    R = quad.R
    vb = R.T @ v
    acc = (R @ np.array([0, 0, thrust]) + R @ (-np.array(af.lin_drag) * vb)) / af.mass - np.array([0, 0, G])
    J = af.J
    om_dot = np.linalg.solve(J, torque - np.cross(om, J @ om) - af.rot_drag * om)
    phi, th, _ = e
    # Euler-angle kinematics (ZYX)
    W = np.array([[1, np.sin(phi) * np.tan(th), np.cos(phi) * np.tan(th)],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi) / np.cos(th), np.cos(phi) / np.cos(th)]])
    return np.concatenate([v, acc, W @ om, om_dot])


def linearise(cfg):
    w_h = np.sqrt(cfg.airframe.mass * G / (4 * cfg.airframe.kf))
    x0 = np.zeros(12)
    u0 = np.full(4, w_h)
    eps = 1e-5
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))
    f0 = plant_derivative(cfg, x0, u0)
    for i in range(12):
        dx = np.zeros(12)
        dx[i] = eps
        A[:, i] = (plant_derivative(cfg, x0 + dx, u0) - f0) / eps
    for i in range(4):
        du = np.zeros(4)
        du[i] = eps * w_h
        B[:, i] = (plant_derivative(cfg, x0, u0 + du) - f0) / (eps * w_h)
    return A, B, w_h


# ------------------------------------------------------------- 2. loop shaping
def attitude_loop_tf(cfg, axis=0):
    """Linear model of one attitude axis with the cascaded P / PID controller."""
    c = cfg.ctrl
    tau = cfg.airframe.motor_tau
    kp, ki, kd = c.rate_p[axis], c.rate_i[axis], c.rate_d[axis]
    ka = c.att_p[axis]
    # rate loop: C_r(s) = kd s + kp + ki/s  (angular-acceleration command, torque = J * cmd)
    # plant from commanded angular acceleration to rate: 1/(tau s + 1) * 1/s
    num_C = [kd, kp, ki]
    den_C = [1, 0]
    num_P = [1]
    den_P = np.polymul([tau, 1], [1, 0])
    L_rate = signal.TransferFunction(np.polymul(num_C, num_P), np.polymul(den_C, den_P))
    # closed inner loop T_r = L/(1+L)
    num_L, den_L = L_rate.num, L_rate.den
    T_num = num_L
    T_den = np.polyadd(den_L, num_L)
    # outer: angle = rate / s, controller ka
    L_att = signal.TransferFunction(ka * T_num, np.polymul(T_den, [1, 0]))
    return L_rate, L_att


def margins(L):
    w = np.logspace(-1, 3, 4000)
    w, mag, phase = signal.bode(L, w)
    mag_lin = 10 ** (mag / 20)
    i_c = np.argmin(np.abs(mag_lin - 1))
    pm = 180 + phase[i_c]
    wc = w[i_c]
    i_180 = np.where(np.diff(np.sign(phase + 180)))[0]
    gm = -mag[i_180[0]] if len(i_180) else np.inf
    return dict(crossover_rad_s=float(wc), phase_margin_deg=float(pm), gain_margin_db=float(gm)), (w, mag, phase)


# --------------------------------------------------------------- 3. step tests
def step_test(cfg, kind, T=4.0):
    """Hover at 5 m, then apply a step. Returns time, response, reference."""
    quad = Quadrotor(cfg.airframe, cfg.battery)
    quad.reset((0, 0, 5.0))
    quad.on_ground = False
    quad.rotor[:] = np.sqrt(cfg.airframe.mass * G / (4 * cfg.airframe.kf))

    class Est:                               # perfect state feedback for the stability study
        pass
    est = Est()
    fc = FlightController(cfg, quad)
    dt = cfg.sim.dt
    ts, ys, refs = [], [], []
    hover_T = cfg.airframe.mass * G
    for k in range(int(T / dt)):
        t = k * dt
        est.R_WB, est.omega, est.pos, est.vel = quad.R, quad.w.copy(), quad.p.copy(), quad.v.copy()
        on = t >= 0.5
        roll, pitch, yaw = rot_to_euler(quad.R)
        if kind == 'roll':
            ref = np.radians(10) if on else 0.0
            sp = {'type': 'attitude', 'roll': ref, 'pitch': 0.0, 'yaw_rate': 0.0, 'thrust': hover_T / np.cos(ref)}
            y = roll
        elif kind == 'pitch':
            ref = np.radians(10) if on else 0.0
            sp = {'type': 'attitude', 'roll': 0.0, 'pitch': ref, 'yaw_rate': 0.0, 'thrust': hover_T / np.cos(ref)}
            y = pitch
        elif kind == 'yaw':
            ref = np.radians(45) if on else 0.0
            sp = {'type': 'position', 'p': np.array([0, 0, 5.0]), 'yaw': ref}
            y = yaw
        elif kind == 'altitude':
            ref = 6.0 if on else 5.0
            sp = {'type': 'position', 'p': np.array([0, 0, ref]), 'yaw': 0.0}
            y = quad.p[2]
        elif kind == 'gust':
            ref = 0.0
            sp = {'type': 'position', 'p': np.array([0, 0, 5.0]), 'yaw': 0.0}
            y = quad.p[0]
        w_cmd = fc.update(sp, est, dt)
        f_ext = np.array([6.0, 0, 0]) if (kind == 'gust' and 0.5 <= t < 1.0) else np.zeros(3)
        quad.step(w_cmd, dt, external_force=f_ext)
        ts.append(t)
        ys.append(y)
        refs.append(ref)
    return np.array(ts), np.array(ys), np.array(refs)


def step_metrics(t, y, ref, t0=0.5):
    m = t >= t0
    t, y, ref = t[m] - t0, y[m], ref[m]
    y0 = y[0]
    final = ref[-1]
    amp = final - y0
    if abs(amp) < 1e-9:
        return {}
    yn = (y - y0) / amp
    rise = t[np.argmax(yn >= 0.9)] - t[np.argmax(yn >= 0.1)]
    overshoot = max(0.0, (yn.max() - 1) * 100)
    outside = np.where(np.abs(yn - 1) > 0.05)[0]
    settle = t[outside[-1]] if len(outside) else 0.0
    return dict(rise_time_s=float(rise), overshoot_pct=float(overshoot), settling_time_5pct_s=float(settle),
                steady_state_error=float(abs(1 - yn[-1])))


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    cfg = Config()
    out = {}

    # 1. open loop
    A, B, w_h = linearise(cfg)
    eig = np.linalg.eigvals(A)
    out['hover_rotor_speed_rad_s'] = float(w_h)
    out['open_loop_eigenvalues'] = sorted([[float(e.real), float(e.imag)] for e in eig])
    ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(12)])
    out['controllable'] = bool(np.linalg.matrix_rank(ctrb, tol=1e-6) == 12)

    # 2. loop margins
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (axis, name) in enumerate([(0, 'roll/pitch'), (2, 'yaw')]):
        L_rate, L_att = attitude_loop_tf(cfg, axis)
        mr, (w, mag, ph) = margins(L_rate)
        ma, (w2, mag2, ph2) = margins(L_att)
        out[f'{name}_rate_loop'] = mr
        out[f'{name}_attitude_loop'] = ma
        axes[0, col].semilogx(w, mag, label='rate loop')
        axes[0, col].semilogx(w2, mag2, label='attitude loop')
        axes[0, col].axhline(0, color='k', lw=0.6)
        axes[0, col].set_ylabel('|L| [dB]')
        axes[0, col].set_title(f'{name} axis: PM rate {mr["phase_margin_deg"]:.0f} deg, '
                               f'PM att {ma["phase_margin_deg"]:.0f} deg')
        axes[0, col].legend()
        axes[1, col].semilogx(w, ph)
        axes[1, col].semilogx(w2, ph2)
        axes[1, col].axhline(-180, color='k', lw=0.6)
        axes[1, col].set_ylabel('phase [deg]')
        axes[1, col].set_xlabel('frequency [rad/s]')
        for a in axes[:, col]:
            a.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'stability_bode.png'), dpi=120)

    # closed-loop pole migration vs attitude gain (stability boundary)
    gains = np.linspace(0.5, 200, 200)
    fig, ax = plt.subplots(figsize=(6, 5))
    unstable_from = None
    for g in gains:
        c2 = Config()
        c2.ctrl.att_p = (g, g, c2.ctrl.att_p[2])
        _, L = attitude_loop_tf(c2, 0)
        poles = np.roots(np.polyadd(L.den, L.num))
        ax.plot(poles.real, poles.imag, '.', color=plt.cm.viridis(g / gains.max()), ms=4)
        if unstable_from is None and poles.real.max() > 0:
            unstable_from = float(g)
    _, L = attitude_loop_tf(cfg, 0)
    p0 = np.roots(np.polyadd(L.den, L.num))
    ax.plot(p0.real, p0.imag, 'rx', ms=10, mew=2, label=f'design (K_att={cfg.ctrl.att_p[0]})')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlim(-60, 15)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title("Roll closed-loop poles vs attitude gain 0.5..200")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'stability_root_locus.png'), dpi=120)
    out['roll_attitude_gain_unstable_from'] = unstable_from
    out['design_closed_loop_poles_roll'] = sorted([[float(p.real), float(p.imag)] for p in p0])

    # 3. step responses on the nonlinear model
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    titles = {'roll': 'Roll 10 deg (lateral)', 'pitch': 'Pitch 10 deg (longitudinal)',
              'yaw': 'Yaw 45 deg (directional)', 'altitude': 'Altitude +1 m', 'gust': '6 N gust for 0.5 s (x)'}
    for ax, kind in zip(axes, titles):
        t, y, ref = step_test(cfg, kind)
        scale = np.degrees if kind in ('roll', 'pitch', 'yaw') else (lambda a: a)
        ax.plot(t, scale(ref), 'k--', lw=1, label='reference')
        ax.plot(t, scale(y), lw=2, color='#2a6fdb', label='response')
        ax.set_title(titles[kind])
        ax.set_xlabel('t [s]')
        ax.grid(alpha=0.3)
        if kind == 'gust':
            out['gust'] = dict(max_deviation_m=float(np.abs(y).max()),
                               recovered_within_10cm_s=float(t[np.where(np.abs(y) > 0.1)[0][-1]] - 0.5)
                               if (np.abs(y) > 0.1).any() else 0.0)
        else:
            out[f'step_{kind}'] = step_metrics(t, y, ref)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'stability_steps.png'), dpi=120)

    with open(os.path.join(RES, 'stability.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'open_loop_eigenvalues'}, indent=2))
    print('open-loop eigenvalues (real parts):', np.round(sorted(eig.real), 3))


if __name__ == '__main__':
    main()
