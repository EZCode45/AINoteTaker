import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import re
path = 'comparisons_validation.csv'

df = pl.read_csv(path)
print(df.head(5))
print(df.null_count())
print(df['split'].unique())
print(df['batch'].unique())
df = df.with_columns(df['info'].map_elements(eval).alias('infodict'))
df = df.with_columns(df['infodict'].map_elements(lambda x: x['post']).alias('post'))
# print(df['infodict'][0])
print(df['post'][0])
df = df.with_columns(df['summaries'].map_elements(lambda x: eval(re.sub(r"\}\s+\{","},{", x))).alias('summarieslist'))
print(df['summarieslist'][0])

features = df['post']
targets = df[]