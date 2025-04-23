import pandas as pd

# Load the filtered dataset
data = pd.read_csv("datasets/filtered_disease_dataset.csv")

# Get unique diseases
diseases = data['disease'].unique()

# Sort and print
print("✅ Filtered Diseases in Dataset:")
for disease in sorted(diseases):
    print("-", disease)
