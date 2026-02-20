import pandas as pd

# File paths
csv1_path = "/home/usluesyr/ai_image_detector/data/real/test/labeled_real_images.csv"
csv2_path = "/home/usluesyr/ai_image_detector/data/images_meta_desciption.csv"
output_path = "/home/usluesyr/ai_image_detector/data/images_meta_desciption2.csv"

# Columns from csv1 to add to csv2
columns_to_add = ["category", "scene", "description"]

# Read CSVs
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Update or add columns without overwriting existing data
for col in columns_to_add:
    # Create a mapping from filename to value
    map_series = df1.set_index('filename')[col]
    
    # Only fill missing values in CSV2
    if col in df2.columns:
        df2[col] = df2[col].fillna(df2['filename'].map(map_series))
    else:
        df2[col] = df2['filename'].map(map_series)

# Save updated CSV
df2.to_csv(output_path, index=False)

print(f"CSV2 updated without overwriting existing data and saved to {output_path}")