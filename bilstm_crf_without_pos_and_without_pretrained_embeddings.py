import pandas as pd
import numpy as np
import random
import string
import spacy
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import keras

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", ""))
        out.append(out_i)
    return out

def get_recall_precision(test_labels,pred_labels,tag):
    flat_test_labels = [item for sublist in test_labels for item in sublist]
    flat_pred_labels = [item for sublist in pred_labels for item in sublist]
    
    test_indices = set([i for i, x in enumerate(flat_test_labels) if x == tag])
    pred_indices = set([i for i, x in enumerate(flat_pred_labels) if x == tag])
    
    recall = len(test_indices.intersection(pred_indices))/len(test_indices)
    precision = len(test_indices.intersection(pred_indices))/len(pred_indices)
    f_score = 2* precision * recall / (precision + recall)
    return recall, precision, f_score


if __name__ == "__main__":
    pickle_in = open("training_and_testing_data.pickle", "rb")
    data_dict = pickle.load(pickle_in)
    X_tr = data_dict["X_tr"]
    X_te = data_dict["X_te"]
    y_tr = data_dict["y_tr"]
    y_te = data_dict["y_te"]
    max_len = 75
    n_words = data_dict["n_words"]
    n_tags = data_dict["n_tags"]
    tag2idx = data_dict["tag2idx"]
    pos2idx = data_dict["pos2idx"]
    word2idx = data_dict["word2idx"]
    ## Model definition
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20,
                      input_length=max_len)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    print(model.summary())
    history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=100,
                    validation_split=0.1, verbose=1)
    #Testing
    test_pred = model.predict(X_te, verbose=1)
    idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels = pred2label(test_pred)
    test_labels = pred2label(y_te)
    print("Recall, Precision and F-score are",
          get_recall_precision(test_labels, pred_labels, "Destination"))
    model.save("BILSTM+CRF_without_pos_without_embeddings")
    