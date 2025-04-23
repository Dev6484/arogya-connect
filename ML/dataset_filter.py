import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/kapilkumar/Developer/GitHub/PBL_ML_Work/new disease/datasets/improved_disease_dataset.csv')

# List of diseases to exclude
diseases_to_exclude = [
    "AIDS", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Alcoholic hepatitis", "Cancer", "HIV", "Vertigo"
]

# Apply the filter to exclude rows where the disease is in the exclusion list
filtered_data = data[~data['disease'].isin(diseases_to_exclude)]

# Check if filtering works correctly
filtered_diseases = filtered_data['disease'].unique()

# Print filtered diseases
print("Filtered Diseases in Dataset (After Exclusion):")
for disease in filtered_diseases:
    print(disease)

# Optionally, save the filtered data to a new CSV
filtered_data.to_csv('/Users/kapilkumar/Developer/GitHub/PBL_ML_Work/new disease/datasets/filtered_disease_dataset.csv', index=False)
