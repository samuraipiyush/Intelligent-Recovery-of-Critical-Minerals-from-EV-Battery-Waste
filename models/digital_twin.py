"""
Digital twin of an EV battery recycling process.

Physical pipeline:
  Leaching reactor → Separation unit → Metal recovery unit

Mass balance ODE per metal i:
  dM_i/dt = F_in * c_i - k_leach(T) * M_i   (leaching reactor)

Temperature-dependent rate constant via Arrhenius:
  k(T) = A * exp(-Ea / (R * T))

After leaching the dissolved fraction passes through:
  - Separation unit  (efficiency eta_sep)
  - Metal recovery   (efficiency eta_rec)
"""

import numpy as np
from scipy.integrate import solve_ivp
from models.composition import compute_composition
from models.recovery import recovery_efficiency as BASE_EFF

# Arrhenius parameters (illustrative, scaled for typical hydrometallurgical leaching)
_A = 1e6       # pre-exponential factor (1/min)
_Ea = 50_000   # activation energy (J/mol)
_R = 8.314     # gas constant (J/mol·K)

# Separation and recovery unit efficiencies (static, adjustable via sliders)
_ETA_SEP = 0.92
_ETA_REC = 0.95


def _k_leach(T_celsius: float) -> float:
    T_K = T_celsius + 273.15
    return _A * np.exp(-_Ea / (_R * T_K))


def run_simulation(
    battery_type: str,
    mass_kg: float,
    feed_rate_kg_per_min: float,
    temperature_c: float,
    efficiency_scale: float = 1.0,
    t_end_min: float = 120.0,
    n_points: int = 300,
) -> dict:
    """
    Simulate time-dependent recovery for each metal in the battery.

    Args:
        battery_type:        One of NMC, LFP, LCO.
        mass_kg:             Total batch mass fed to the reactor (kg).
        feed_rate_kg_per_min: Rate at which black mass enters leaching reactor.
        temperature_c:       Reactor temperature in °C.
        efficiency_scale:    Multiplier on base recovery efficiency [0.5 – 1.5].
        t_end_min:           Total simulation time (minutes).
        n_points:            Number of time points to evaluate.

    Returns:
        Dict with keys:
          time       – 1-D array of time values (min)
          leached    – {metal: array of cumulative leached mass (kg)}
          recovered  – {metal: array of cumulative finally recovered mass (kg)}
          final      – {metal: final recovered mass (kg)}
    """
    comp = compute_composition(battery_type, mass_kg)
    k = _k_leach(temperature_c)
    t_span = (0.0, t_end_min)
    t_eval = np.linspace(0.0, t_end_min, n_points)

    metals = list(comp.keys())
    n = len(metals)

    # State vector: M_i(t) = mass of metal i still un-leached in the reactor (kg)
    M0 = np.array([comp[m] for m in metals])

    # Per-metal leaching rate can differ slightly; we scale by base efficiency
    eff_vec = np.array(
        [BASE_EFF.get(m, 0.7) * efficiency_scale for m in metals]
    )

    def odes(t, M):
        # Simple first-order leaching: dM/dt = -k * eta * M
        return -k * eff_vec * M

    sol = solve_ivp(odes, t_span, M0, t_eval=t_eval, method="RK45", dense_output=False)

    time = sol.t
    leached = {}
    recovered_over_time = {}

    for idx, metal in enumerate(metals):
        M_reactor = sol.y[idx]
        leached_mass = M0[idx] - M_reactor          # cumulative leached from reactor
        final_recovered = leached_mass * _ETA_SEP * _ETA_REC
        leached[metal] = leached_mass
        recovered_over_time[metal] = final_recovered

    final = {m: float(recovered_over_time[m][-1]) for m in metals}

    return {
        "time": time,
        "leached": leached,
        "recovered": recovered_over_time,
        "final": final,
        "composition": comp,
    }


def sensitivity_over_temperature(
    battery_type: str,
    mass_kg: float,
    feed_rate_kg_per_min: float,
    temp_range: tuple[float, float] = (40.0, 90.0),
    n_temps: int = 10,
) -> dict:
    """Return final recovery vs temperature for each metal."""
    temps = np.linspace(temp_range[0], temp_range[1], n_temps)
    results: dict[str, list] = {}

    for T in temps:
        sim = run_simulation(battery_type, mass_kg, feed_rate_kg_per_min, T)
        for metal, val in sim["final"].items():
            results.setdefault(metal, []).append(val)

    return {"temperatures": list(temps), "final_per_temp": results}
