battery_data = {
    "NMC": {"Li": 0.07, "Co": 0.15, "Ni": 0.2, "Mn": 0.1},
    "LFP": {"Li": 0.05, "Fe": 0.35, "P": 0.15},
    "LCO": {"Li": 0.07, "Co": 0.6}
}

def compute_composition(battery_type, mass):
    comp = battery_data[battery_type]
    return {metal: frac * mass for metal, frac in comp.items()}