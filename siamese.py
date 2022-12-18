import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense
import tensorflow as tf

farzi = pd.read_csv('neg_examples.csv')

farzi['name'] = [x[:20] for x in farzi['name']]
farzi['matched_name'] = [x[:20] for x in farzi['matched_name']]

farzi = farzi.reset_index()
farzi = farzi.drop(['index'], axis=1)

diction = {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
           'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24,
           'y': 25, 'z': 26, '&': 27, '(': 28, ')': 28, '-': 29, '0': 30, '1': 31, '2': 32, '3': 33, '4': 34, '5': 35,
           '6': 36, '7': 37, '8': 38, '9': 39, '_': 40}

def new_array(name):
    an_array = []
    for i in name:
        for key, value in diction.items():  # for name, age in dictionary.iteritems()
            if i == key:
                an_array.append(int(value))

    an_array = np.array([an_array])

    shape = np.shape(an_array)
    padded_array = np.zeros((20))
    padded_array[:shape[1]] = an_array
    return padded_array

farzi['name_array'] = farzi['name'].apply(new_array)
farzi['matched_name_array'] = farzi['matched_name'].apply(new_array)

farzi['name_array'] = farzi['name_array'].apply(np.int64)
farzi['matched_name_array'] = farzi['matched_name_array'].apply(np.int64)

def create_array(pairs):
    list_pairs = []
    for i in pairs:
        list_temp = []
        for j in i:
            list_temp.append(list(j))
        list_pairs.append(list_temp)

    pairs = np.array(list_pairs)
    return pairs

threshold = 0.5

def round_up(a):
    if a - int(a) >= threshold:
        return int(a) + 1
    return int(a)

train, test = train_test_split(farzi, train_size=0.8)

max_words = 41
embedding_dim = 100

X = train[["name_array", "matched_name_array"]].to_numpy()
y = train[["label"]].to_numpy()

kf = StratifiedKFold(n_splits=5)
kf.get_n_splits(X)
print(kf)

fold_no = 1
for train_index, valid_index in kf.split(X, y):
    print("TRAIN:", train_index, "VALIDATION:", valid_index)
    pairs_train, pairs_valid = X[train_index], X[valid_index]
    label_train, label_valid = y[train_index], y[valid_index]

    lstm_layer = tf.keras.layers.LSTM(units=32, dropout=0.2, recurrent_dropout=0.2)

    emb = tf.keras.layers.Embedding(max_words, embedding_dim, trainable=True)

    input1 = tf.keras.Input(shape=(20,))
    e1 = emb(input1)
    e1 = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(e1)
    x1 = lstm_layer(e1)

    input2 = tf.keras.Input(shape=(20,))
    e2 = emb(input2)
    e2 = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(e2)
    x2 = lstm_layer(e2)

    mhd = lambda x: tf.keras.backend.abs(x[0] - x[1])
    merged = tf.keras.layers.Lambda(function=mhd, output_shape=lambda x: x[0], name='L2_distance')([x1, x2])
    preds = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    model = tf.keras.Model(inputs=[input1, input2], outputs=preds)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy",
                                                                         tf.keras.metrics.Precision(),
                                                                         tf.keras.metrics.Recall()])
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='siamese/model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs1'),
    ]

    pairs_train = create_array(pairs_train)
    pairs_valid = create_array(pairs_valid)
    label_train = label_train.reshape(len(label_train), )
    label_valid = label_valid.reshape(len(label_valid), )

    model.fit([pairs_train[:, 0], pairs_train[:, 1]], label_train, batch_size=64, epochs=20, verbose=1,
              validation_data=([pairs_valid[:, 0], pairs_valid[:, 1]], label_valid), callbacks=my_callbacks)  #

    y_pred1 = model.predict([pairs_valid[:, 0], pairs_valid[:, 1]])
    y_pred1 = y_pred1.reshape((len(label_valid),))

    vfunc = np.vectorize(round_up)
    x1 = vfunc(y_pred1)
    print("x1: ", x1)

    unique1, counts1 = np.unique(x1, return_counts=True)
    print("prediction: ", dict(zip(unique1, counts1)))
    print("Accuracy: ", accuracy_score(label_valid, x1))
    print("Precision: ", precision_score(label_valid, x1))
    print("Recall: ", recall_score(label_valid, x1))

    fold_no = fold_no + 1

print(model.summary())

pairs_test = test[["name_array", "matched_name_array"]].to_numpy()
label_test = test[["label"]].to_numpy()

pairs_test = create_array(pairs_test)
label_test = label_test.reshape(len(label_test), 1)

y_pred1 = model.predict([pairs_test[:, 0], pairs_test[:, 1]])
y_pred1 = y_pred1.reshape((len(label_test),))

vfunc = np.vectorize(round_up)
x1 = vfunc(y_pred1)
print("x1: ", x1)

unique1, counts1 = np.unique(x1, return_counts=True)
print("prediction: ", dict(zip(unique1, counts1)))
print("Accuracy: ", accuracy_score(label_test, x1))
print("Precision: ", precision_score(label_test, x1))
print("Recall: ", recall_score(label_test, x1))

def check(name1, name2):
    ip1 = new_array(name1)
    ip2 = new_array(name2)
    ip1 = ip1.astype(int)
    ip2 = ip2.astype(int)
    ip1 = ip1.reshape((1, 20))
    ip2 = ip2.reshape((1, 20))
    return model.predict([ip1, ip2])