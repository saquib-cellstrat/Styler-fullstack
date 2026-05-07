def interpolate(a: float, b: float, t: float) -> float:
    clamped_t = min(1.0, max(0.0, t))
    return a + (b - a) * clamped_t
