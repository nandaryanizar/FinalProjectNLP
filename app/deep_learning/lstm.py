import numpy as np
import logging
import os

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.models import load_model
from sklearn import metrics

class LSTM:
    train_seq_X = None
    test_seq_X = None
    train_Y = None
    test_Y = None
    hidden_state = 300
    batch_size = 64

    classifier = None

    def __init__(self, train_X, train_Y, test_X, test_Y):
        embedding_index = {}
        for i, line in enumerate(open('glove.6B/glove.6B.100d.txt')):
            values = line.split()
            embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')

        # Create tokenizer object
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(train_X)
        word_index = tokenizer.word_index

        # Convert text to padded sequence of tokens and load previous model if available, disable train method
        self.test_seq_X = sequence.pad_sequences(tokenizer.texts_to_sequences(test_X), maxlen=70)
        if os.path.isfile('lstm.h5'):
            self.classifier = load_model('lstm.h5')
            return

        # Save if no previous model loaded
        self.train_seq_X = sequence.pad_sequences(tokenizer.texts_to_sequences(train_X), maxlen=70)
        self.train_Y = train_Y
        self.test_Y = test_Y

        if os.path.isfile('lstm.h5'):
            self.classifier = load_model('lstm.h5')
            return

        # Create word embeddings mapping
        embedding_matrix = np.zeros((len(word_index) + 1), 300)
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Creating layer
        # Add input layer
        input_layer = layers.Input((70, ))

        # Add the word embedding layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add LSTM layer
        lstm_layer = layers.LSTM(self.hidden_state)(embedding_layer)

        # Output layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        self.classifier = model
              
        logging.info("LSTM model created")

    # def preprocessing(self, train_X, test_X):
    def train(self):
        if self.train_seq_X is None:
            return

        self.classifier.fit(self.train_seq_X, self.train_Y, batch_size=self.batch_size)

        predictions = self.classifier.predict(self.test_seq_X)

        predictions = predictions.argmax(axis=-1)

        self.classifier.save('lstm.h5')

        return metrics.accuracy_score(predictions, self.test_Y)

    def predict(self):
        if not os.path.isfile('lstm.h5') or self.train_seq_X is None:
            self.train()
        
        predictions = self.classifier.predict(self.test_Y)

        return predictions.argmax(axis=-1), predictions
        