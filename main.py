import polars as pl
from sklearn.model_selection import train_test_split

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['train'])

print(df.head())