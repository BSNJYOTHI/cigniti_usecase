#This is the usecase of cigniti for automating the defects rised

project dictionary

Data folder -> initial data
Final_Data.xlsx -> Data used for training purpose
labeled_data_lsa.csv -> Data used for predicting reasons on invalid defects

data_acquisition.ipynb -> it modifies the input json data to get the desired training data

EDA&ModelTrainer.ipynb -> This file handles EDA, text preprocessing, stopwords removal, vectorization, identification of best model
lsa.ipynb, topic_comments.ipynb,  kmeans.ipynb -> all these files has code for finding the ways to get reason for invalid defects and finalized lsa model for defect reason identification
train_lsa_comments.ipynb -> training code for invalid defet reasons.

defect_model.pkl -> serialized model for defect classifier
reasons_for_invalid_defect_model.pkl -> serialized model for reason identifier

main.py -> It creats an API endpoint for our ML application using fastapi
app.py -> It creates frontend for our ML application using streamlit library and also using fastapi created as backend
