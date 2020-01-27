# destination-ner
This repo implemented methods to find destination in user utterances


#Details of files
Following code files are important

* prepare_training_testing_sets.py - File contains code to generate training and testing sets from DummyData_augmented.txt . DummyData_augmented.txt is provided separately. This files reads data line by line. Tags destinations using list of entities known in the dataset. It replaces some of the entities in sequences to "Unknown" tag. It also replaces 10% non-destination tag words to "Unknown"
* bilstm_crf_without_pos_and_without_pretrained_embeddings.py -  File contains code to train a Bi-LSTM and CRF based NER model. It doesn't use POS tags or pretrained embeddings as input to model
* bilstm_crf_with_pos_and_without_pretrained_embeddings.py -  File contains code to train a Bi-LSTM and CRF based NER model. It uses POS tags but doesn't use pretrained embeddings as input to model




#Models used

#Deployment

#Tests

#Process followed
