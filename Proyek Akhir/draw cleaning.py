import pandas as pd

df = pd.read_csv("clean_data/English_Football_2018-2023_With_Form.csv")
df = df[df["Winner"] != 1]
df["Winner"] = df["Winner"].replace(2, 1)
df.to_csv("clean_data/no_draws.csv", index=False)
print("Cleaned data saved to 'no_draws.csv'.")
