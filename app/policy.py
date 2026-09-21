def get_risk_band(probability: float) -> str:
    """
    Optional Configurable Policy Layer.
    Maps a calibrated default probability to a demonstration risk band.
    This does NOT represent a real credit decision.
    """
    if probability < 0.10:
        return "LOW"
    elif probability < 0.25:
        return "MODERATE"
    elif probability < 0.50:
        return "HIGH"
    else:
        return "VERY_HIGH"
