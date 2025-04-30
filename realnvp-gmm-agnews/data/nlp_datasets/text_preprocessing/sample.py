import pandas as pd

# Step 1: Load the CSV (no header)
df = pd.read_csv("C:\\Users\\Hardy\\Desktop\\flowgmm-public\\data\\nlp_datasets\\test.csv", header=None, names=['label', 'text'])

# Step 2: Make sure the label column is treated correctly
df['label'] = df['label'].astype(int)

# Step 3: Sample 80 rows for each label
sampled_df = (
    df.groupby('label')
    .apply(lambda x: x.sample(n=80, random_state=42))
    .reset_index(drop=True)
)

# Step 4: Save the sampled data (if needed)
sampled_df.to_csv('sampled_test.csv', index=False, header=False)

print("Sampling completed. 80 samples per class.")
