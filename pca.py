import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
from sklearn.decomposition import IncrementalPCA  # Incremental PCA for dimensionality reduction
from sklearn.preprocessing import StandardScaler  # StandardScaler for data normalization
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier for classification
from sklearn.model_selection import train_test_split  # Utility for splitting data into training and testing sets
from sklearn.metrics import accuracy_score, classification_report  # Metrics to evaluate model performance
import glob  # Module to handle file path patterns
from tqdm import tqdm  # tqdm for displaying progress bars
import matplotlib.pyplot as plt  # Matplotlib for plotting
import os  # Module for operating system dependent functionality
import time  # Module for measuring time

# Define the dataset path with a wildcard to load all CSV files in the directory
dataset_path = '/Users/d3vil/Desktop/NBaIOT/*.csv'
# Retrieve the list of all file paths matching the pattern
all_files_in_dataset = glob.glob(dataset_path)

# Read the first file to extract the column names (assuming all files have the same structure)
example_df = pd.read_csv(all_files_in_dataset[0])
feature_names = example_df.columns  # Store column names for later use

# Define the batch size for processing files
dataset_batch_size = 1000
# Define the number of principal components to reduce to
n_components = 5

# Initialize lists to hold the feature data and target labels
X_data = []
y_data = []

print("\nInitiating batch loading...\n")
# Variable to keep track of the total number of loaded rows
total_loaded_rows = 0

# Loop over the dataset files in batches
for i in tqdm(range(0, len(all_files_in_dataset), dataset_batch_size), desc="Loading and scaling batches"):
    # Get the current batch of files
    dataset_batch_files = all_files_in_dataset[i: i + dataset_batch_size]
    print(f"\nProcessing batch {i} to {i + dataset_batch_size}\n")  

    # List to hold data for the current batch
    dataset_batch_data = []

    # Process each file in the batch
    for f in dataset_batch_files:
        print(f"Reading {f}") 
        # Read the CSV file into a DataFrame
        df = pd.read_csv(f)
        
        # Extract target label from the filename assuming the format 'name.target.extension'
        individual_filename = os.path.basename(f)
        filename_parts = individual_filename.split('.')
        target = filename_parts[1] if len(filename_parts) == 3 else f"{filename_parts[1]}_{filename_parts[2]}"
        print(f"Extracted target '{target}' from file '{f}'")
   
        # Convert DataFrame to numpy array and add to the batch data
        features = df.values
        dataset_batch_data.append(features)
        # Extend target labels list with the current target for each row in the DataFrame
        y_data.extend([target] * len(df))

    if dataset_batch_data:
        print(f"\nConcatenating {len(dataset_batch_data)} dataframes\n")
        # Concatenate all data in the current batch into a single numpy array
        batch_combined_data = np.vstack(dataset_batch_data)
        total_loaded_rows += len(batch_combined_data)

        print(f"Batch {i}-{i+dataset_batch_size}: Features shape: {batch_combined_data.shape}")

        # Append the combined batch data to the main dataset list
        X_data.append(batch_combined_data)

# Combine all batch data into a single numpy array
X_data = np.vstack(X_data)
# Convert target labels list to a numpy array
y_data = np.array(y_data)
# Calculate the original data size in bytes
original_data_size = X_data.nbytes
print(f"\nOriginal dataset size: {original_data_size} bytes, Total rows loaded: {total_loaded_rows}") 

# Display the unique target labels and their counts
unique_targets, counts = np.unique(y_data, return_counts=True)
print(f"\nUnique targets and their counts: {dict(zip(unique_targets, counts))}")

# Split the dataset into training and testing sets with stratified sampling to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"\nFirst few rows of X_train: {X_train[:5]}")
print(f"\nFirst few rows of X_test: {X_test[:5]}")
print(f"\nFirst few targets of y_train: {y_train[:5]}")
print(f"\nFirst few targets of y_test: {y_test[:5]}")

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nX_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"\nFirst few rows of X_train_scaled: {X_train_scaled[:5]}")
print(f"\nFirst few rows of X_test_scaled: {X_test_scaled[:5]}\n")

# Initialize Incremental PCA with the desired number of components
ipca = IncrementalPCA(n_components=n_components)

# Fit the Incremental PCA on the training data in batches
for i in tqdm(range(0, X_train_scaled.shape[0], dataset_batch_size), desc="Incremental PCA Fitting"):
    batch = X_train_scaled[i: i + dataset_batch_size]  
    print(f"Batch shape for PCA: {batch.shape}")
    ipca.partial_fit(batch)

print("\nInitializing PCA transformation of training data...")
# Transform the training data to the lower-dimensional space
X_train_pca = ipca.transform(X_train_scaled)
print("\nInitializing PCA transformation of test data...")
# Transform the test data to the lower-dimensional space
X_test_pca = ipca.transform(X_test_scaled)

print(f"\nX_train_pca shape: {X_train_pca.shape}")
print(f"\nX_test_pca shape: {X_test_pca.shape}")
print(f"\nFirst few rows of X_train_pca: {X_train_pca[:5]}")
print(f"\nFirst few rows of X_test_pca: {X_test_pca[:5]}")

# Calculate the size of the PCA-reduced dataset in bytes
pca_size = X_train_pca.nbytes + X_test_pca.nbytes  
# Calculate and display the data size reduction percentage
print(f"\nPCA-reduced dataset size: {pca_size} bytes")
print(f"\nData size reduction: {((original_data_size - pca_size) / original_data_size) * 100:.2f}%")

print("\nTraining original model...")

print(f"\nX_train_scaled shape: {X_train_scaled.shape}")
print(f"\nFirst few rows of X_train_scaled: {X_train_scaled[:5]}")
print(f"\ny_train shape: {y_train.shape}")
print(f"\nFirst few targets of y_train: {y_train[:5]}")

# Initialize RandomForestClassifier for the original (non-PCA) dataset
clf_rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
print("\nFitting the original model with scaled training data...")
# Record the time taken to fit the model
clf_original_start_time = time.time()
clf_rf_original.fit(X_train_scaled, y_train)
clf_original_end_time = time.time()
print(f"\nOriginal model training completed in {clf_original_end_time - clf_original_start_time:.2f} seconds")

# Display feature importances if available
if hasattr(clf_rf_original, 'feature_importances_'):
    print("Feature importances of the original model:")
    print(clf_rf_original.feature_importances_)

print("\nTraining PCA model...")

print(f"X_train_pca shape: {X_train_pca.shape}")
print(f"First few rows of X_train_pca: {X_train_pca[:5]}")
print(f"y_train shape: {y_train.shape}")
print(f"First few targets of y_train: {y_train[:5]}")

# Initialize RandomForestClassifier for the PCA-reduced dataset
clf_rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
print(f"\nFitting the PCA model with PCA-transformed training data...")
# Record the time taken to fit the model
clf_pca_start_time = time.time()
clf_rf_pca.fit(X_train_pca, y_train)
clf_pca_end_time = time.time()
print(f"\nPCA model training completed in {clf_pca_end_time - clf_pca_start_time:.2f} seconds")

# Display feature importances if available
if hasattr(clf_rf_pca, 'feature_importances_'):
    print("\nFeature importances of the PCA model:")
    print(clf_rf_pca.feature_importances_)

print("\nGenerating predictions for original model and PCA model...")
# Make predictions using the original and PCA models
y_pred_original = clf_rf_original.predict(X_test_scaled)
y_pred_pca = clf_rf_pca.predict(X_test_pca)

# Calculate the accuracy for both models
accuracy_original = accuracy_score(y_test, y_pred_original)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print("\n--- Original Data Results ---")
print(f"\nAccuracy(Original): {accuracy_original}")
print("\nClassification Report(Original):\n", classification_report(y_test, y_pred_original))

print("\n--- PCA-Reduced Data Results ---")
print(f"\nAccuracy(PCA-Reduced): {accuracy_pca}")
print("\nClassification Report(PCA-Reduced):\n", classification_report(y_test, y_pred_pca))

print(f"\nPCA components used: {n_components}.")

print("\nTop features contributing to each PCA component:")
# Display the top contributing features for each PCA component
for component_num, component in enumerate(ipca.components_):
    top_feature_indices = component.argsort()[-5:][::-1]  # Indices of the top 5 features
    top_feature_names = [feature_names[i] for i in top_feature_indices]  # Get the feature names
    print(f"Component {component_num}: {top_feature_names}")

# Define column names for the PCA-reduced dataset
pca_column_names = [f"PC{i+1}" for i in range(n_components)]
# Create a DataFrame for the PCA-reduced dataset with the target column
reduced_df = pd.DataFrame(X_train_pca, columns=pca_column_names)
reduced_df['target'] = y_train

# Save the PCA-reduced dataset to a CSV file
reduced_df.to_csv('/Users/d3vil/Desktop/NBaIOT/pca_reduced_dataset.csv', index=False)
