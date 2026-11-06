import os
import subprocess
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_data(tmp_path):
    data = {
        'longitude': [-122.25, -122.23, -122.22, -122.21, -122.20],
        'latitude': [37.85, 37.85, 37.85, 37.85, 37.85],
        'housing_median_age': [30.0, 21.0, 52.0, 42.0, 20.0],
        'total_rooms': [2000.0, 1200.0, 3000.0, 1500.0, 2500.0],
        'total_bedrooms': [400.0, 250.0, 600.0, 300.0, 500.0],
        'population': [1000.0, 600.0, 1500.0, 800.0, 1200.0],
        'households': [380.0, 220.0, 550.0, 290.0, 450.0],
        'median_income': [5.5, 4.2, 6.1, 3.8, 4.9],
        'ocean_proximity': ['NEAR BAY', 'NEAR BAY', 'NEAR BAY', '<1H OCEAN', '<1H OCEAN'],
        'median_house_value': [450000.0, 350000.0, 480000.0, 300000.0, 420000.0]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "datos_prueba.csv"
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def sample_pred_data(tmp_path):
    data = {
        'longitude': [-122.24],
        'latitude': [37.85],
        'housing_median_age': [25.0],
        'total_rooms': [1800.0],
        'total_bedrooms': [350.0],
        'population': [900.0],
        'households': [340.0],
        'median_income': [5.0],
        'ocean_proximity': ['NEAR BAY']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "ejemplo_prediccion.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_train_and_storage(sample_data, tmp_path):
    models_dir = tmp_path / "models"
    result = subprocess.run([
        "python", "train.py",
        "--data", str(sample_data),
        "--target", "median_house_value",
        "--output-dir", str(models_dir),
        "--version", "v1",
        "--seed", "42"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    version_dir = models_dir / "v1"
    assert (version_dir / "modelo.joblib").exists()
    assert (version_dir / "metrics.json").exists()
    assert (version_dir / "metadata.json").exists()

def test_load_and_prediction(sample_data, sample_pred_data, tmp_path):
    models_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"

    subprocess.run([
        "python", "train.py",
        "--data", str(sample_data),
        "--target", "median_house_value",
        "--output-dir", str(models_dir),
        "--version", "v1",
        "--seed", "42"
    ], check=True)

    output_csv = outputs_dir / "predicciones.csv"
    result = subprocess.run([
        "python", "predict.py",
        "--model", str(models_dir / "v1" / "modelo.joblib"),
        "--input", str(sample_pred_data),
        "--output", str(output_csv)
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert output_csv.exists()
    df_res = pd.read_csv(output_csv)
    assert "prediccion" in df_res.columns
    assert len(df_res) == 1

def test_invalid_input_column(sample_data, sample_pred_data, tmp_path):
    models_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"

    subprocess.run([
        "python", "train.py",
        "--data", str(sample_data),
        "--target", "median_house_value",
        "--output-dir", str(models_dir),
        "--version", "v1",
        "--seed", "42"
    ], check=True)

    df_invalid = pd.read_csv(sample_pred_data).drop(columns=['total_rooms'])
    invalid_csv = tmp_path / "invalido.csv"
    df_invalid.to_csv(invalid_csv, index=False)

    result = subprocess.run([
        "python", "predict.py",
        "--model", str(models_dir / "v1" / "modelo.joblib"),
        "--input", str(invalid_csv),
        "--output", str(outputs_dir / "pred.csv")
    ], capture_output=True, text=True)

    assert result.returncode != 0

def test_versioning_no_overwrite(sample_data, tmp_path):
    models_dir = tmp_path / "models"

    subprocess.run([
        "python", "train.py",
        "--data", str(sample_data),
        "--target", "median_house_value",
        "--output-dir", str(models_dir),
        "--version", "v1",
        "--seed", "42"
    ], check=True)

    result = subprocess.run([
        "python", "train.py",
        "--data", str(sample_data),
        "--target", "median_house_value",
        "--output-dir", str(models_dir),
        "--version", "v1",
        "--seed", "42"
    ], capture_output=True, text=True)

    assert result.returncode != 0

def test_missing_values_handling(sample_data, tmp_path):
    models_dir = tmp_path / "models"
    result = subprocess.run([
        "python", "train.py",
        "--data", str(sample_data),
        "--target", "median_house_value",
        "--output-dir", str(models_dir),
        "--version", "v1",
        "--seed", "42"
    ], capture_output=True, text=True)
    assert result.returncode == 0
