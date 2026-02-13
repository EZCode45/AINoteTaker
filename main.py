import polars as pl
from sklearn.model_selection import train_test_split
import tensorflow as tf

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['train'])

train_df, val_df = train_test_split(df.to_pandas(), test_size=0.2, random_state=42)
model = tf.keras.Model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_df, validation_data=val_df, epochs=5)
model.evaluate(val_df)

