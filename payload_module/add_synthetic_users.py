
import pandas as pd
from pathlib import Path

# Dataset location
input_path = Path(
    r"D:\MAJOR PROJECT DATA\payload_generated\FADE_like.csv"
)

output_path = input_path.parent / "FADE_like_with_users.csv"

# Load dataset
df = pd.read_csv(input_path)

# Reproducible synthetic user assignment
number_of_users = 100
user_ids = [
    f"USER_{i:03d}"
    for i in range(1, number_of_users + 1)
]

# Distribute samples across synthetic users
df["user"] = [
    user_ids[i % number_of_users]
    for i in range(len(df))
]

# Save as a new file
df.to_csv(output_path, index=False)

print("Synthetic user column added.")
print(f"Output file: {output_path}")
print("\nSamples per user:")
print(df["user"].value_counts().sort_index())

print("\nPreview:")
print(df[["sample", "class", "user"]].head(10))
