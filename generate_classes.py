import pandas as pd
import json
import os

# Define file paths
csv_file_path = "artifacts/data_ingestion/data/Training_set.csv"  # Update the correct path
json_output_path = "artifacts/data_ingestion/data/class_mapping.json"

# Load CSV file
df = pd.read_csv(csv_file_path)

# Extract unique labels (assuming 'label' column exists)
unique_labels = sorted(df['label'].unique())  # Sorting ensures consistency

# Create mapping dictionary (label -> index)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Save as JSON file
with open(json_output_path, "w") as json_file:
    json.dump(label_to_index, json_file, indent=4)

print(f"✅ Class mapping JSON saved at: {json_output_path}")