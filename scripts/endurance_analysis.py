"""Modules 1 & 7 - batteries, propellers, altering speed and increasing flight time.

(a) Hover power: momentum (actuator-disc) theory vs the simulator's motor model.
(b) Power, endurance and range vs cruise speed:
      * analytical forward-flight momentum theory (Glauert induced velocity,
        parasite drag, profile power) - shows the classic "bucket" curve with a
        best-endurance and a best-range speed
      * measured on the closed-loop simulator (straight flights at fixed speed)
(c) Flight time vs battery capacity when the extra cells' mass is accounted for
    (diminishing returns).
Outputs docs/figures/endurance.png and results/endurance.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from cyclops.config import Config, G  # noqa: E402
from cyclops.sim.simulator import Simulation  # noqa: E402
from cyclops.world.world import World  # noqa: E402

RHO = 1.225
FIG = os.path.join(ROOT, 'docs', 'figures')
RES = os.path.join(ROOT, 'results')
USABLE = 0.8          # only 80 % of the pack is used (LiPo health / reserve)
FM = 0.65             # rotor figure of merit
CDA = 0.04            # m^2, parasite drag area of the airframe
PACK_WH_PER_KG = 150  # LiPo specific energy


def usable_wh(cfg, capacity_mah=None):
    cap = capacity_mah or cfg.battery.capacity_mah
    return cfg.battery.nominal_v * cap / 1000 * USABLE


def forward_flight_power(cfg, V, mass=None):
    """Electrical power [W] from momentum theory in forward flight."""
    m = mass or cfg.airframe.mass
    A = 4 * np.pi * (cfg.airframe.prop_diameter / 2) ** 2
    W = m * G
    D = 0.5 * RHO * CDA * V ** 2 + np.mean(cfg.airframe.lin_drag[:2]) * V
    T = np.hypot(W, D)
    alpha = np.arctan2(D, W)                       # disc tilt
    vh = np.sqrt(T / (2 * RHO * A))
    vi = vh
    for _ in range(100):                           # Glauert fixed point
        vi = T / (2 * RHO * A * np.sqrt((V * np.cos(alpha)) ** 2 + (V * np.sin(alpha) + vi) ** 2))
    p_induced = T * vi / FM
    p_parasite = T * V * np.sin(alpha)
    p_profile = 0.05 * T * vh * (1 + 4.65 * (V / 60) ** 2)  # small blade profile term
    shaft = p_induced + p_parasite + p_profile
    return shaft / cfg.battery.eta_esc_motor + cfg.battery.avionics_w


def sim_power(cfg, V, T_meas=6.0):
    """Mean electrical power in steady straight flight at speed V on the full simulator."""
    c = Config()
    c.planner.cruise_speed = V if V > 0 else 0.01
    c.planner.max_speed = max(V, 0.5)
    c.ctrl.max_vel_xy = max(V + 1, 2)
    w = World([], [0, 0, 2.0], [500, 0, 2.0])
    sim = Simulation(w, 'baseline', cfg=c)
    sim.arm_and_takeoff()
    while sim.cmd.mode != 'MISSION':
        sim.step()
    t0 = sim.t
    while sim.t < t0 + 3.0:                       # accelerate
        sim.step()
    e0, t1 = sim.quad.battery.energy_used_wh, sim.t
    while sim.t < t1 + T_meas:
        sim.step()
    return (sim.quad.battery.energy_used_wh - e0) * 3600 / (sim.t - t1)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    cfg = Config()
    af = cfg.airframe
    A = 4 * np.pi * (af.prop_diameter / 2) ** 2
    W = af.mass * G
    p_ideal = W ** 1.5 / np.sqrt(2 * RHO * A)
    out = dict(disc_area_m2=A, disc_loading_N_m2=W / A, hover_ideal_power_W=p_ideal,
               hover_power_momentum_W=forward_flight_power(cfg, 0.0),
               usable_energy_Wh=usable_wh(cfg))

    speeds = np.linspace(0, 12, 49)
    P_th = np.array([forward_flight_power(cfg, v) for v in speeds])
    E = usable_wh(cfg)
    endur = E / P_th * 60
    rng = speeds * E / P_th * 3.6
    i_e, i_r = np.argmax(endur), np.argmax(rng)
    out.update(best_endurance_speed=float(speeds[i_e]), max_endurance_min=float(endur[i_e]),
               best_range_speed=float(speeds[i_r]), max_range_km=float(rng[i_r]),
               hover_endurance_min=float(endur[0]))

    sim_speeds = [0, 2, 4, 6, 8]
    P_sim = [sim_power(cfg, v) for v in sim_speeds]
    out['sim_power_W'] = dict(zip([str(s) for s in sim_speeds], P_sim))
    out['sim_hover_endurance_min'] = E / P_sim[0] * 60

    caps = np.arange(2000, 16001, 1000)
    pack0 = cfg.battery.nominal_v * cfg.battery.capacity_mah / 1000 / PACK_WH_PER_KG
    t_cap = []
    for c in caps:
        m = af.mass - pack0 + cfg.battery.nominal_v * c / 1000 / PACK_WH_PER_KG
        if m * G > 0.5 * af.max_thrust:            # keep T/W >= 2 for control authority
            t_cap.append(np.nan)
            continue
        t_cap.append(usable_wh(cfg, c) / forward_flight_power(cfg, 0.0, mass=m) * 60)
    out['capacity_sweep'] = {str(int(c)): (None if np.isnan(t) else float(t)) for c, t in zip(caps, t_cap)}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(speeds, P_th, lw=2, color='#2a6fdb', label='momentum theory')
    ax[0].plot(sim_speeds, P_sim, 'o', color='#e8833a', ms=7, label='simulator (no translational lift)')
    ax[0].set_xlabel('cruise speed [m/s]')
    ax[0].set_ylabel('electrical power [W]')
    ax[0].set_title('Power required')
    ax[0].legend()
    ax[1].plot(speeds, endur, lw=2, color='#3fa45b', label='endurance [min]')
    ax[1].axvline(speeds[i_e], color='#3fa45b', ls=':')
    ax1b = ax[1].twinx()
    ax1b.plot(speeds, rng, lw=2, color='#8a5cf5', label='range [km]')
    ax1b.axvline(speeds[i_r], color='#8a5cf5', ls=':')
    ax[1].set_xlabel('cruise speed [m/s]')
    ax[1].set_ylabel('endurance [min]', color='#3fa45b')
    ax1b.set_ylabel('range [km]', color='#8a5cf5')
    ax[1].set_title(f'best endurance {speeds[i_e]:.1f} m/s, best range {speeds[i_r]:.1f} m/s')
    ax[2].plot(caps / 1000, t_cap, 'o-', lw=2, color='#d1495b')
    ax[2].axvline(cfg.battery.capacity_mah / 1000, color='k', ls=':', label='current 4S 5 Ah')
    ax[2].set_xlabel('4S pack capacity [Ah]')
    ax[2].set_ylabel('hover flight time [min]')
    ax[2].set_title('Bigger battery = more mass (diminishing returns)')
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'endurance.png'), dpi=120)
    with open(os.path.join(RES, 'endurance.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
