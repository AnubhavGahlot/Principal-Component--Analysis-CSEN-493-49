IoT Data Reduction using PCA

This research highlights the use of Principal Component Analysis (PCA) to reduce the dimensionality of IoT data while retaining its critical properties. The Incremental PCA (IPCA) technique is used to handle huge datasets efficiently. The research also compares the performance of models trained on original data with PCA-reduced data.

 Table of Contents

- [Introduction](introduction)
- [Dataset](dataset)
- [Prerequisites](prerequisites)
- [Usage](usage)
- [Research Structure](research-structure)
- [Results](results)
- [Acknowledgements](acknowledgements)

 Introduction

The growth of the Internet of Things (IoT) has resulted in an explosion in the volume of data created by linked devices. Real-time analytics, machine learning, and autonomous decision-making rely heavily on effective data management. This research investigates the use of PCA to minimize IoT data sizes, resulting in more effective data handling without sacrificing accuracy.

 Dataset

The N-BaIoT dataset, which contains network traffic information from 9 IoT devices, is used in the research. Every CSV file in the dataset is associated with a specific device and attack type. 
Source URL: 
https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset?resource=download
https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot 

 Prerequisites

Before running the code, ensure you have the following software and libraries installed:

- Python 3.6 or higher
- pandas
- numpy
- scikit-learn
- tqdm
- matplotlib
- glob

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn tqdm matplotlib
```

 Usage

1. Download the Dataset: Download the N-BaIoT dataset and place the CSV files in a directory. Update the `dataset_path` variable in the code to point to this directory.

2. Run the Scriptresearch: Execute the script to preprocess the data, apply Incremental PCA, and compare the performance of RandomForest classifiers trained on the original and PCA-reduced data.

```bash
python3 iot_pca_data_reduction.py
```

 Research Structure

- `iot_pca_data_reduction.py`: This is the main script for loading, preprocessing, reducing, and evaluating IoT data using Incremental PCA.
- `README.md`: This file provides an overview and instructions for running the project.
- `docs/`: This directory contains additional documentation and notes on the project.

 Results

The script outputs the following:

- The original and PCA-reduced dataset sizes.
- Data size reduction percentage.
- Performance metrics (accuracy, precision, recall, F1-score) for models trained on original and PCA-reduced data.
- Feature importances for both models.
- Top features contributing to each PCA component.

 Acknowledgements

- The N-BaIoT dataset, which was used in this research, was created to investigate the features of botnets in IoT environments. The dataset offers insightful information about how Internet of Things devices behave in different attack scenarios.


