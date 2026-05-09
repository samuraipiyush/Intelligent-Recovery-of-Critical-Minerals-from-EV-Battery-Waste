recovery_efficiency = {
    "Li": 0.8,
    "Co": 0.9,
    "Ni": 0.85,
    "Mn": 0.7,
    "Fe": 0.8,
    "P": 0.75
}

def compute_recovery(comp):
    return {
        metal: comp[metal] * recovery_efficiency.get(metal, 0.7)
        for metal in comp
    }