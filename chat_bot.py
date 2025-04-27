#import model
from tensorflow.keras.models import load_model
import preprocess
import jread
import numpy as np

text=input("Enter your text here")

model = load_model(r'C:\Users\HP\NLP\chat.h5')

pro_text=preprocess.stop_word(text)

def sentence_to_vector(sentence, vocab):
    pro_text=preprocess.stop_word(text) 
    bag = [1 if word in pro_text else 0 for word in vocab]
    return (np.array(bag))[np.newaxis,:]

model_input=sentence_to_vector(text,preprocess.pvocab)

index = np.argmax(model.predict(model_input))  # Predict the class index

if index<5:
    print(jread.j_data["intents"][0]["responses"])


elif index<10:
    print(jread.j_data["intents"][1]["responses"])

elif index<14:
    print(jread.j_data["intents"][2]["responses"])

elif index<19:
    print(jread.j_data["intents"][3]["responses"])
