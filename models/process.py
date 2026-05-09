def select_process(battery_type, recovered):
    # rules
    if battery_type == "NMC":
        return "Hydrometallurgy", "High-value metals (Co, Ni) → selective recovery preferred"
    
    elif battery_type == "LCO":
        return "Hydrometallurgy", "High cobalt content → high selectivity needed"
    
    elif battery_type == "LFP":
        return "Pyrometallurgy", "Lower-value metals → bulk processing preferred"
    
    else:
        return "Pyrometallurgy", "Default choice for mixed feed"