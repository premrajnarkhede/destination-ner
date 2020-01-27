import pandas as pd
import numpy as np
import random
import string
import spacy
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_letters+string.digits
    rand_int = random.randint(3,11)
    string_random = [random.choice(letters) for i in range(stringLength)]
    chunk_array = []
    for chunk in chunks(string_random,rand_int):
        chunk_array.append("".join(chunk))
    return chunk_array




if __name__=="__main__":
    data = open("dummyData_augmented.txt").readlines()
    nlp = spacy.load("en_core_web_sm")
    array = []
    destination_entities =set(["XYYZ","home","work","office","parents'","home","place","my","kid's","school","grocery","store","76","Charlottenstrasse","coffee","shop","pizzeria","starbucks","police","station","book","Jalgaon","45","schutzenstrasse","railway","'s","3100","North","Main","St.","23","alexanderplatz","Friedrichstrasse","KFC","Kid","kids","kid","cafe","gas","parents","children","'"])

    for textindex,text in enumerate(data):
        doc = nlp(text.strip())
        for token in doc:
            text_items = [token.text]
            if token.text == "XYYZ":
                tag = "Destination"
                text_items = ["Unknown"]
            elif token.text in destination_entities:
                tag = "Destination"
            else:
                tag = ""
                rand_int = random.randint(1,11)
                if rand_int==3:
                    text_items = ["Unknown"]
            for item in text_items:
                print(item,tag)
                array.append(dict(Sentence=textindex,Word=item,POS=token.pos_,Tag=tag))

    df = pd.DataFrame(array) 
    words = list(set(df["Word"].values))
    words.append("ENDPAD")
    words.append("Unknown")
    n_words = len(words); n_words
    tags = list(set(df["Tag"].values))
    n_tags = len(tags); n_tags
    pos = list(set(df["POS"].values))
    pos.append("BLANK")
    pos_tags = len(pos)
    print(n_tags,pos_tags)
    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                       s["POS"].values.tolist(),
                                                       s["Tag"].values.tolist())]
    grouped = df.groupby("Sentence").apply(agg_func)
    sentences = [s for s in grouped]
    max_len = 75
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    pos2idx = {p: i for i, p in enumerate(pos)}
    X = [[word2idx.get(w[0],word2idx["Unknown"]) for w in s] for s in sentences]
    X_pos = [[pos2idx.get(w[1]) for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)
    X_pos = pad_sequences(maxlen=max_len, sequences=X_pos, padding="post", value=pos2idx["BLANK"])
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx[""])
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx[""])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    X_tr, X_te, y_tr, y_te, X_pos_tr, X_pos_te = train_test_split(X, y, X_pos, test_size=0.2)
    
    data_dict = {}
    data_dict["X_tr"] = X_tr
    data_dict["X_te"] = X_te
    data_dict["y_tr"] = y_tr
    data_dict["y_te"] = y_te
    data_dict["X_pos_tr"] = X_pos_tr
    data_dict["X_pos_te"] = X_pos_te
    data_dict["n_words"] = n_words
    data_dict["n_tags"] = n_tags
    data_dict["words"] = words
    data_dict["tags"] = tags
    data_dict["pos"] = pos
    data_dict["pos_tags"] = pos_tags
    data_dict["word2idx"] = word2idx
    data_dict["tag2idx"] = tag2idx
    data_dict["pos2idx"] = pos2idx
    pickle_out = open("training_and_testing_data.pickle", "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    