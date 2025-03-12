import pandas as pd
import os
import numpy as np

#Set working directory
os.chdir("")

df= pd.read_csv("Toyota.csv", index_col=0, na_values=["###","??","????"])


print("First 5 rows of the dataset: ")
print(df.head())

print("\nData types in each column: ")
print(df.dtypes)

print("\nCount of unique data types: ")
print(df.dtypes.value_counts())

numericData= df.select_dtypes(include=["int64", "float64"])
objectData= df.select_dtypes(include=["object"])
print("\nNumeric Data Columns: ")
print(numericData.head)
print("\nObject Data Columns: ")
print(objectData.head)

print("\nDataFrame Info: ")
print(df.info)

for col in df.columns:
    print(f"\nUnique Values in'{col}':")
    print(df[col].unique())

df["Doors"]= df["Doors"].replace({"three":3, "Four":4, "Five":5}).astype("float64")
df["MetColor"] = df["MetColor"].astype("object")
df["Automatic"] = df["Automatic"].astype("object")

print("\nMissing Values Count in Each Column")
print(df.isnull.sum())

df.to_csv("Toyota_Cleaned.csv", index=False)

print("\nData Cleaning and analysis completed Cleaned data saved as 'Toyota_Cleaned.csv'. ")

