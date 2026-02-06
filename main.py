import polars as pl
from sklearn.model_selection import train_test_split

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['train'])

train_df, val_df = train_test_split(df.to_pandas(), test_size=0.2, random_state=42)
train_df = pl.from_pandas(train_df)
val_df = pl.from_pandas(val_df)
train_df.write_csv('train_split.csv')
val_df.write_csv('val_split.csv')
print("Train and validation splits created and saved as 'train_split.csv' and 'val_split.csv'.")
test_df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['test'])
test_df.write_csv('test_split.csv')
print("Test split saved as 'test_split.csv'.")

