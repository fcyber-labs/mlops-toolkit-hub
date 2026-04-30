def test_metrics_loading_functions():
    """Test that metric loading functions handle missing files gracefully"""
    # Import functions (mock if needed)
    from src.monitoring.dashboard import load_metrics_phase1, load_metrics_phase2

    # Should return dict even if files missing
    metrics1 = load_metrics_phase1()
    metrics2 = load_metrics_phase2()

    assert isinstance(metrics1, dict)
    assert isinstance(metrics2, dict)


def test_drift_calculation_psi():
    """Test PSI calculation works correctly"""
    from src.monitoring.dashboard import calculate_psi

    # Identical distributions
    ref = [1, 2, 3, 4, 5] * 20
    actual = [1, 2, 3, 4, 5] * 20
    psi = calculate_psi(ref, actual)
    assert psi < 0.05  # Should be very small

    # Different distributions
    ref = [1, 2, 3, 4, 5] * 20
    actual = [5, 6, 7, 8, 9] * 20
    psi = calculate_psi(ref, actual)
    assert psi > 0.1  # Should detect drift


def test_drift_status_classification():
    """Test PSI threshold classification"""
    from src.monitoring.dashboard import drift_status

    assert drift_status(0.05) == "🟢 Stable"
    assert drift_status(0.15) == "🟡 Moderate"
    assert drift_status(0.35) == "🔴 Significant"


def test_ks_calculation():
    """Test Kolmogorov-Smirnov test works"""
    from scipy.stats import ks_2samp
    import numpy as np

    # Same distribution
    ref = np.random.normal(0, 1, 100)
    same = np.random.normal(0, 1, 100)
    stat, p = ks_2samp(ref, same)
    assert p > 0.05

    # Different distribution
    diff = np.random.normal(2, 1, 100)
    stat, p = ks_2samp(ref, diff)
    assert p < 0.05
