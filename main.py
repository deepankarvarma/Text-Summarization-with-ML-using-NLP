import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Embedding, RepeatVector, TimeDistributed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 2: Load the data into pandas dataframes
train_data = pd.read_csv('cnn_dailymail/train.csv')
valid_data = pd.read_csv('cnn_dailymail/validation.csv')
test_data = pd.read_csv('cnn_dailymail/test.csv')

# Step 3: Perform any necessary preprocessing on the data
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_data(data):
    data['article'] = data['article'].apply(lambda x: ' '.join([stemmer.stem(word.lower()) for word in word_tokenize(x) if word.lower() not in stop_words]))
    data['highlights'] = data['highlights'].apply(lambda x: ' '.join([stemmer.stem(word.lower()) for word in word_tokenize(x) if word.lower() not in stop_words]))
    return data

train_data = preprocess_data(train_data)
valid_data = preprocess_data(valid_data)
test_data = preprocess_data(test_data)

# Step 4: Split the data into input and target variables
train_x = train_data['article'].values
train_y = train_data['highlights'].values
valid_x = valid_data['article'].values
valid_y = valid_data['highlights'].values
test_x = test_data['article'].values
test_y = test_data['highlights'].values

# Step 5: Encode the input and target data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
train_x = tokenizer.texts_to_sequences(train_x)
valid_x = tokenizer.texts_to_sequences(valid_x)
test_x = tokenizer.texts_to_sequences(test_x)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_y)
train_y = tokenizer.texts_to_sequences(train_y)
valid_y = tokenizer.texts_to_sequences(valid_y)
test_y = tokenizer.texts_to_sequences(test_y)

# Step 6: Define and train the neural network model
vocab_size = len(tokenizer.word_index) + 1
max_length_x = max([len(seq) for seq in train_x])
max_length_y = max([len(seq) for seq in train_y])

def define_model():
    model = Sequential()
    model.add(Input(shape=(max_length_x,)))
    model.add(Embedding(vocab_size, 256))
    model.add(LSTM(256))
    model.add(RepeatVector(max_length_y))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

model = define_model()
model.summary()

model.fit(train_x, np.array(train_y), epochs=50, validation_data=(valid_x, np.array(valid_y)))
model.save('text_summarization.h5')
# Step 7: Evaluate the model on the validation data
preds = model.predict(valid_x)
preds = np.argmax(preds, axis=-1)

accuracy = accuracy_score(np.hstack(valid_y), np.hstack(preds))
precision = precision_score(np.hstack(valid_y), np.hstack(preds), average='macro')
recall = recall_score(np.hstack(valid_y), np.hstack(preds), average='macro')
f1 = f1_score(np.hstack(valid_y), np.hstack(preds), average='macro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 8: Generate highlights for the test data using the trained model
test_x_encoded = tokenizer.texts_to_sequences(test_data['article'].values)
test_x_encoded = pad_sequences(test_x_encoded, maxlen=max_length_x, padding='post')

preds = model.predict(test_x_encoded)
preds = np.argmax(preds, axis=-1)

pred_highlights = []
for pred in preds:
    highlight = [tokenizer.index_word[word_idx] for word_idx in pred if word_idx > 0]
    pred_highlights.append(' '.join(highlight))

test_data['predicted_highlights'] = pred_highlights

# Step 9: Print the test data with predicted highlights
print(test_data[['article', 'highlights', 'predicted_highlights']])
