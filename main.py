import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pl.read_csv('hf://datasets/lytang/MeetingBank-transcript/' + splits['train'])
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(
    df['source'],
    df['type'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
x_train_vz = vectorizer.fit_transform(x_train)
x_test_vz = vectorizer.transform(x_test)

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.fit_transform(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train_vz.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_vz.toarray(), y_train_enc, epochs=5, batch_size=32, validation_split=0.1)
y_pred = model.predict(x_test_vz.toarray())
y_pred_classes = y_pred.argmax(axis=1)
print("Accuracy:", accuracy_score(y_test_enc, y_pred_classes))
print("Classification Report:\n", classification_report(y_test_enc, y_pred_classes, target_names=label_encoder.classes_))
