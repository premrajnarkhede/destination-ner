from flask import Flask
from flask import jsonify
import spacy
import pandas as pd
import numpy as np
import random
import string
import spacy
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import keras
from keras_contrib.utils import save_load_utils


app = Flask(__name__)

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", ""))
        out.append(out_i)
    return out


pickle_in = open("training_and_testing_data.pickle", "rb")
data_dict = pickle.load(pickle_in)
X_tr = data_dict["X_tr"]
X_te = data_dict["X_te"]
y_tr = data_dict["y_tr"]
y_te = data_dict["y_te"]
X_pos_tr = data_dict["X_pos_tr"]
X_pos_te = data_dict["X_pos_te"]
max_len = 75
n_words = data_dict["n_words"]
n_tags = data_dict["n_tags"]
tag2idx = data_dict["tag2idx"]
pos2idx = data_dict["pos2idx"]
word2idx = data_dict["word2idx"]
idx2tag = {i: w for w, i in tag2idx.items()}
pos = data_dict["pos"]
## Model definition
input = Input(shape=(max_len,))
word_emb = Embedding(input_dim=n_words + 1, output_dim=20,
input_length=max_len)(input) # 20-dim embedding
pos_input = Input(shape=(max_len,))
pos_emb = Embedding(input_dim= len(pos) , output_dim=10,
input_length=max_len)(pos_input)
modified_input = keras.layers.concatenate([word_emb, pos_emb])
model_1 = Bidirectional(LSTM(units=50, return_sequences=True,
recurrent_dropout=0.1))(modified_input)
model = TimeDistributed(Dense(50, activation="relu"))(model_1)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output
model = Model([input,pos_input], out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.load_weights("BILSTM+CRF_with_pos_without_embeddings.model")

nlp = spacy.load("en_core_web_sm")

# model1 = save_load_utils.load_all_weights('BILSTM+CRF_without_pos_without_embeddings.model')
# model2 = load_model('BILSTM+CRF_with_pos_without_embeddings.model,')
#model = load_model("BILSTM+CRF_with_pos_without_embeddings.model", custom_objects={'CRF': CRF, 'loss':fake_loss })
# (model,filename)
@app.route("/")
def hello():
    return "Hello World!"


@app.route("/model1/<sentence>")
def model1(sentence):
    array = []
    doc = nlp(sentence)
    for token in doc:
        text = token.text.lower()
        print(text)
#         if token.text in entities_words:
#             tag = "Destination"
#         else:
#             tag = ""
        
        array.append(tuple([token.text,token.pos_,""]))
    print(array)
    X = [[word2idx.get(w[0],word2idx["Unknown"]) for w in s] for s in [array]]
    X_pos = [[pos2idx.get(w[1],pos2idx["PRON"]) for w in s] for s in [array]]

    print(X)
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)
    X_pos = pad_sequences(maxlen=max_len, sequences=X_pos, padding="post", value=pos2idx["BLANK"])
    y = [[tag2idx[w[2]] for w in s] for s in [array]]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx[""])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    #print(y)
    test_pred = model.predict([X,X_pos], verbose=1)
    pred_labels = pred2label(test_pred)
    test_labels = pred2label(y)
    entities = []
    entity = []
    for aindex,a in enumerate(array):
        if pred_labels[0][aindex]=="Destination":
            entity.append(a[0])
        if pred_labels[0][aindex]!="Destination":
            if len(entity)>0:
                entities.append(" ".join(entity))
            entity = []
    if len(entity)>0:
        entities.append(" ".join(entity))
    
                
               
    return jsonify({"destination":entities})


if __name__ == "__main__":
    app.run(host="0.0.0.0",port="5005",threaded=False)