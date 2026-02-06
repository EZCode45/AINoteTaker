import polars as pl
from sklearn.model_selection import train_test_split
import tensorflow

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['train'])

train_df, val_df = train_test_split(df.to_pandas(), test_size=0.2, random_state=42)


