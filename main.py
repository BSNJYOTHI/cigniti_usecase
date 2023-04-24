# text preprocessing modules

# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import pickle
import uvicorn
from fastapi import FastAPI 
import string

app = FastAPI(
    title="Defect Classifier Model API",
    description="A simple API that use NLP model to predict the whether a defect is valid or invalid",
    version="0.1",
)

# load the defect model
with open('defect_model.pkl', 'rb') as f:
    model = pickle.load(f)

#load the reason for invalid defect model

with open('reasons_for_invalid_defect_model.pkl', 'rb') as f:
    model_reasons = pickle.load(f)



stop_words = nltk.corpus.stopwords.words("english")
not_words = ['not', 'no', 'nor', 'don', "don't", 'should', "should've", 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
for word in not_words:
    stop_words.remove(word)

#definig custom stopwords list

#creating a list of custom stopwords
newStopwords = ['image','png','thumbnail']
stop_words.extend(newStopwords)

stop_words = [item.lower() for item in stop_words]
stop_words = set(stop_words)

#cleaning the data
#function for preprocessing the data

#function for preprocessing the data

def remove_urls(text):
    new_text = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return new_text

#make all text lower case
def text_lowercase(text):
    return text.lower()

#remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+','',text)
    return result

#remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)

#tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text

#remove stop words
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

#lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text




def  regex_matching(text):
    #remove regex matchings of below
    text = re.sub(r'\[~accountid:.*\]'," ",text)
    text = re.sub(r'\[([^\]]+)\]\.mp4]'," ",text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'," ",text)
    return text

#function to handle all the above functions
def text_cleaning(text):
    text = text_lowercase(text)
    text = regex_matching(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    
    text = ' '.join(text)
    return text



@app.get("/predict-defect")
def predict_defect(description: str):
    """
    A simple function that receive a defect description and predict the defect.
    
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_text = text_cleaning(description)
    
    # perform prediction
    prediction = model.predict([cleaned_text])
    prediction_reason = model_reasons.predict([cleaned_text])
    output = int(prediction[0])
    output_reason = int(prediction_reason[0])
    probas = model.predict_proba([cleaned_text])
    
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    defects = {0: "Invalid", 1: "Valid"}
    defects_reason = {0:"working as expected",1:"reproducible or environment issue",2:"duplicate issue",3:"cancelling defect"}
    if output==0:
        result = {"Predicted Defect Status": defects[output], "Confidence Score": output_probability,"reason prediction":f"Possible reason for defect to be invalid might be due to {defects_reason[output_reason]}"}
    else:
        # show results
        result = {"Predicted Defect Status": defects[output], "Confidence Score": output_probability}

    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)







