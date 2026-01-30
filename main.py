import polars as pl

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['train'])

print(df.head())