import hug
import spacy
import pickle
import os

print(os.getcwd())
print("Loading spacy..")
nlp  = spacy.load('nl_core_news_lg') 


print("Loading model..")
# load pickled model
with open("./model/random_forest_clf.pkl", "rb") as f:
    clf = pickle.load(f)

def argmax(array):
      return array.index(max(array))
    
@hug.get('/predict')
def predict(text : hug.types.shorter_than(500,convert=hug.types.text)):
    print("Making prediction on " + text)
    doc = nlp(text)
    
    validation_data = [(token.text, token.vector) for token in doc if token.pos_ == "NOUN"]

    if len(validation_data) > 0:
    
        word_list   = [v[0] for v in validation_data]
        
        vector_list = [v[1] for v in validation_data]
        
        
        # make prediction with pure-predict object
        predictions = clf.predict_proba(vector_list)

        json_result_list = []
        prediction_text = ['de','het']
        for prediction, word in zip(predictions,word_list):
            json_result = {}
            json_result['woord'] = word
            highest_proba_index = argmax(prediction)
            highest_proba = prediction[highest_proba_index]
            highest_proba_name = prediction_text[highest_proba_index]
            json_result['predicition'] = {highest_proba_name : highest_proba }
            json_result_list.append(json_result)
        return {'msg':'success', 'predictions' : json_result_list }
    else:
        return {'msg':'fail','detail': 'Cannot find any Nouns in the given sentence(s)'}


