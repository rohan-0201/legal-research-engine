import pandas as pd

df = pd.read_csv("formatted_tax_cases.csv")

section_fields = [
    "Facts", "Issues", "PetArg", "RespArg",
    "Section", "Precedent", "CDiscource", "Conclusion"
]

df_filtered = df.dropna(subset=section_fields, how='all')

df_filtered.fillna('', inplace=True)

df_filtered.reset_index(drop=True, inplace=True)

df_filtered.to_csv("filtered_tax_cases.csv", index=False)

print(f"Filtered dataset saved to 'filtered_tax_cases.csv'")
print(f"🧹 Original rows: {len(df)}, Filtered rows: {len(df_filtered)}")