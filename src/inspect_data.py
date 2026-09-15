from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
data_path = project_folder / "data" / "raw" / "students.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print("----------------------------")
print("Rows:", df.shape)
print("Columns:", df.shape)

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFirst five rows:")
print(df.head())

print("\nCategorical value counts:")
for column in df.select_dtypes(include="object").columns:
    print(f"\n{column}")
    print(df[column].value_counts())
