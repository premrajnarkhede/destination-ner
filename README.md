# destination-ner
This repo implemented methods to find destination in user utterances


## Details of files
Following code files are important

* prepare_training_testing_sets.py - File contains code to generate training and testing sets from DummyData_augmented.txt . DummyData_augmented.txt is provided separately. This files reads data line by line. Tags destinations using list of entities known in the dataset. It replaces some of the entities in sequences to "Unknown" tag. It also replaces 10% non-destination tag words to "Unknown". It splits sequences into training and testing sets. Results are stored to pickle file.
* bilstm_crf_without_pos_and_without_pretrained_embeddings.py -  File contains code to train a Bi-LSTM and CRF based NER model. It doesn't use POS tags or pretrained embeddings as input to model. It reads pre-formatted features from pickle, declared keras model, calculates recall, precision and f-score and stores model to file.
* bilstm_crf_with_pos_and_without_pretrained_embeddings.py -  File contains code to train a Bi-LSTM and CRF based NER model. It uses POS tags but doesn't use pretrained embeddings as input to model. It reads pre-formatted features from pickle, declared keras model, calculates recall, precision and f-score and stores model to file.
* app.py Simple flask based app to serve results of models trained\

## Models used
BiLSTM + CRF based model was chosen for this demo as RNN + CRF combination was giving good results in many combinations in literature available. https://nlpprogress.com/english/named_entity_recognition.html . Bi-LSTM is not too expensive to train for smaller datasets and seemed to be go to solution for every NLP problem before transformer architectures arrrived. However transformer architectures are very expensive to train and operate and may not be suitable for applications requiring fast responses.

1) First model doesn't use POS tags or pretrained word vector model : It generates its own 20 dimension embedding while training. It used 50 units for LSTM cells. This is primarily based on https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/

Model summary is as follows:

      Model: "model_1"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      input_1 (InputLayer)         (None, 75)                0         
      _________________________________________________________________
      embedding_1 (Embedding)      (None, 75, 20)            1860      
      _________________________________________________________________
      bidirectional_1 (Bidirection (None, 75, 100)           28400     
      _________________________________________________________________
      time_distributed_1 (TimeDist (None, 75, 50)            5050      
      _________________________________________________________________
      crf_1 (CRF)                  (None, 75, 2)             110       
      =================================================================
      Total params: 35,420
      Trainable params: 35,420
      Non-trainable params: 0
      _________________________________________________________________

This model had tendancy to mark any unseen words as Destination. 

2) Second model uses POS tags. POS tags have their own 10 dimension embeddings while rest of the details remained same.


            Model: "model_2"
            __________________________________________________________________________________________________
            Layer (type)                    Output Shape         Param #     Connected to                     
            ==================================================================================================
            input_1 (InputLayer)            (None, 75)           0                                            
            __________________________________________________________________________________________________
            input_2 (InputLayer)            (None, 75)           0                                            
            __________________________________________________________________________________________________
            embedding_1 (Embedding)         (None, 75, 20)       1860        input_1[0][0]                    
            __________________________________________________________________________________________________
            embedding_2 (Embedding)         (None, 75, 10)       150         input_2[0][0]                    
            __________________________________________________________________________________________________
            concatenate_1 (Concatenate)     (None, 75, 30)       0           embedding_1[0][0]                
                                                                             embedding_2[0][0]                
            __________________________________________________________________________________________________
            bidirectional_1 (Bidirectional) (None, 75, 100)      32400       concatenate_1[0][0]              
            __________________________________________________________________________________________________
            time_distributed_1 (TimeDistrib (None, 75, 50)       5050        bidirectional_1[0][0]            
            __________________________________________________________________________________________________
            crf_1 (CRF)                     (None, 75, 2)        110         time_distributed_1[0][0]         
            ==================================================================================================
            Total params: 39,570
            Trainable params: 39,570
            Non-trainable params: 0
            __________________________________________________________________________________________________



3) Third model was using POS tags as well as pretrained Glove vectors. However it ended up giving results poorer than even model1. It required more fine tuning and hence not described here. You can see model structure in python notebook "With pretrained embeddings and pos tags.ipynb"

## Evaluation

Models were evaluated on parameters recall, precision of only "Destination" tagged entities.

Model 1 : Recall:90.66 Precision:98.5 F-score: 94.41

Model 2 : Recall:93.73 Precision:100 F-score: 96.76


## Deployment and outputs
Model 2 is temporarily deployed at http://157.245.231.247:5005/model1/

Usage send the sentence as parameter after "model1/"

e.g.http://157.245.231.247:5005/model1/find%20me%20a%20police%20station%20quickly

gets output {"destination":["police station"]}



** Please convert ? to %3F in sentence
** Pass the context as a part of sentence itself e.g. Where do you want to to go%3F Alexanderplatz

text: Navigate me to Munich
{"destination":["Munich"]}

text: Take me to 34 Torstrasse
{"destination":["34 Torstrasse"]}

text: Destination my work
{"destination":["my work"]}

text: Drive me to my kids' play
{"destination":["my kids ' play"]}

text:"Find neareast starbucks and drive me there"
{"destination":["starbucks","there"]}

text:"Where do you want to go? Please dont bother"
{"destination":[]}

text:"Nearest book shop"
{"destination":["book shop"]}

text:"Where do you want to go? Checkpoint Charlie"
{"destination":["Checkpoint charlie"]}

text:"find me a "parking place" in "berlin mitte""
{"destination":["place","berlin mitte"]}











## Process followed for training data enrichment

Open source NLU datasets provided were scanned but it would have needed some automation to pickup some relevant dialogue exchanges.

Many new phrases were discovered using following urls

https://owner.ford.com/support/how-tos/sync-technology/sync-3/navigation/using-navigation-voice-commands-with-sync-3.html
https://www.johnhiesterchevy.com/Navigation
https://android.gadgethacks.com/how-to/navigate-hands-free-using-voice-commands-0166496/
https://www.waze.com/forum/viewtopic.php?p=1312171

