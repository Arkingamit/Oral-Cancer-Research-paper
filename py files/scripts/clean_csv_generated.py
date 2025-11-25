import pandas as pd

files = {
    "train": "datasets/triplet/train.csv",
    "val": "datasets/triplet/val.csv",
    "test": "datasets/triplet/test.csv"
}

for name, path in files.items():
    try:
        df = pd.read_csv(path)  # Let pandas auto-detect separator
        print(f"{name} file columns: {df.columns.tolist()}")

        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()

        # Handle BOM if present
        if '\ufeffcase_id' in df.columns:
            df = df.rename(columns={'\ufeffcase_id': 'case_id'})

        # Select only the needed columns
        df_cleaned = df[["case_id", "case_id_pos", "case_id_neg"]]
        output_path = f"datasets/triplet/{name}_cleaned.csv"
        df_cleaned.to_csv(output_path, index=False)
        print(f"?? Cleaned {name} file saved to {output_path}")

    except Exception as e:
        print(f"? Error processing {name}: {e}")
