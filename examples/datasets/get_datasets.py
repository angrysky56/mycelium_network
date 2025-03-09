#!/usr/bin/env python3
"""
Download and prepare common datasets for testing machine learning algorithms.
"""

import os
import csv
import urllib.request

def download_iris():
    """Download the Iris dataset and save it as a CSV file."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    output_path = os.path.join(os.path.dirname(__file__), "iris.csv")
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Iris dataset already exists at {output_path}")
        return output_path
    
    # Download the dataset
    print(f"Downloading Iris dataset from {url}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Iris dataset saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading Iris dataset: {e}")
        return None

def download_wine():
    """Download the Wine dataset and save it as a CSV file."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    output_path = os.path.join(os.path.dirname(__file__), "wine.csv")
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Wine dataset already exists at {output_path}")
        return output_path
    
    # Download the dataset
    print(f"Downloading Wine dataset from {url}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Wine dataset saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading Wine dataset: {e}")
        return None

def download_breast_cancer():
    """Download the Breast Cancer Wisconsin dataset and save it as a CSV file."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    output_path = os.path.join(os.path.dirname(__file__), "breast_cancer.csv")
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Breast Cancer dataset already exists at {output_path}")
        return output_path
    
    # Download the dataset
    print(f"Downloading Breast Cancer dataset from {url}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Breast Cancer dataset saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading Breast Cancer dataset: {e}")
        return None

def main():
    """Download all datasets."""
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    iris_path = download_iris()
    wine_path = download_wine()
    breast_cancer_path = download_breast_cancer()
    
    print("\nDataset Summary:")
    print("---------------")
    if iris_path:
        print(f"Iris dataset: {iris_path}")
    if wine_path:
        print(f"Wine dataset: {wine_path}")
    if breast_cancer_path:
        print(f"Breast Cancer dataset: {breast_cancer_path}")

if __name__ == "__main__":
    main()
