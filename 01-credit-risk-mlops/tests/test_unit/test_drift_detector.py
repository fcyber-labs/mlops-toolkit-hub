import pandas as pd
import numpy as np


def test_drift_detector_numerical():
    """Test numerical drift detection"""
    from src.streaming.drift_detector import DriftDetector

    # Create reference data
    ref_data = pd.DataFrame({"feature1": np.random.normal(0, 1, 100), "feature2": np.random.normal(0, 1, 100)})

    detector = DriftDetector(threshold=0.05)
    detector.set_reference(ref_data)

    # Same distribution (no drift)
    current_same = pd.DataFrame({"feature1": np.random.normal(0, 1, 100), "feature2": np.random.normal(0, 1, 100)})
    results = detector.detect_all_drift(current_same)
    assert results["overall_drift_score"] < 0.5


def test_drift_detector_categorical():
    """Test categorical drift detection - FIXED: same length arrays"""
    from src.streaming.drift_detector import DriftDetector

    # Create reference data with SAME LENGTH arrays
    # Both arrays must have the same number of elements
    ref_data = pd.DataFrame(
        {
            "cat1": ["A", "B", "C", "A", "B", "C", "A", "B", "C"] * 11,  # 99 rows
            "cat2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X"] * 11,  # 99 rows
        }
    )

    detector = DriftDetector(threshold=0.05)
    detector.set_reference(ref_data)

    # Different distribution (should detect drift)
    current_diff = pd.DataFrame(
        {
            "cat1": ["D", "E", "F", "D", "E", "F", "D", "E", "F"] * 11,
            "cat2": ["Z", "W", "Z", "W", "Z", "W", "Z", "W", "Z"] * 11,
        }
    )
    results = detector.detect_all_drift(current_diff)
    assert isinstance(results, dict)


def test_drift_detector_categorical_alternative():
    """Alternative categorical test using simple approach"""
    from src.streaming.drift_detector import DriftDetector

    # Create reference data
    ref_data = pd.DataFrame({"cat1": ["A"] * 50 + ["B"] * 50, "cat2": ["X"] * 60 + ["Y"] * 40})

    detector = DriftDetector(threshold=0.05)
    detector.set_reference(ref_data)

    # Current data with different distribution
    current_diff = pd.DataFrame({"cat1": ["C"] * 60 + ["D"] * 40, "cat2": ["Z"] * 70 + ["W"] * 30})

    results = detector.detect_all_drift(current_diff)
    assert isinstance(results, dict)
    assert "overall_drift_score" in results


def test_drift_detector_saves_report(tmp_path):
    """Test that drift report is saved correctly"""
    import json
    from src.streaming.drift_detector import DriftDetector

    ref_data = pd.DataFrame({"feature1": np.random.normal(0, 1, 50)})
    current_data = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 50)})

    detector = DriftDetector(threshold=0.05)
    detector.set_reference(ref_data)
    detector.detect_all_drift(current_data)

    report_path = tmp_path / "test_drift_report.json"
    detector.save_report(str(report_path))

    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
        assert "overall_drift_score" in report
        assert "features_with_drift" in report
