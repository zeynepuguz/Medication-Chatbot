import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/train.csv")

print("İlk veri boyutu:", df.shape)

df = df[["Question", "Answer"]]

df.columns = ["question", "answer"]

df = df.dropna()

df = df.drop_duplicates()

print("Temizlenmiş veri boyutu:", df.shape)

# Eğitim / doğrulama / test böl
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# Klasör oluşturma
os.makedirs("data/processed", exist_ok=True)

# Kaydetme
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Preprocessing tamamlandı.")