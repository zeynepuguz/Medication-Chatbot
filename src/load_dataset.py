from datasets import load_dataset
import pandas as pd
import os

print("Dataset indiriliyor...")

dataset = load_dataset("truehealth/medicationqa")

print(dataset)

os.makedirs("data/raw", exist_ok=True)

for split in dataset.keys():
    df = pd.DataFrame(dataset[split])

    file_path = f"data/raw/{split}.csv"

    df.to_csv(file_path, index=False)

    print(f"{split} kaydedildi -> {file_path}")

print("İşlem tamamlandı.")