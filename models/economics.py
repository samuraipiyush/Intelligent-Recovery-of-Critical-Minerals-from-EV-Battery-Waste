prices = {
    "Li": 5000,
    "Co": 30000,
    "Ni": 2000,
    "Mn": 500,
    "Fe": 100,
    "P": 200
}

def compute_revenue(recovered):
    revenue_breakdown = {
        metal: recovered[metal] * prices.get(metal, 0)
        for metal in recovered
    }
    total_revenue = sum(revenue_breakdown.values())
    return total_revenue, revenue_breakdown


def compute_profit(revenue, cost):
    return revenue - cost