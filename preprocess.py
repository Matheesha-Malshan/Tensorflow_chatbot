import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import re
from jread import j_data

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lower_casing(content):
    return content.lower()

def remove_punctuation(content):
    return re.sub(r'[^\w\s]', '', lower_casing(content))

def lem(content):
    lr = remove_punctuation(content)
    lr_s = lr.split(" ")
    l = ([lemmatizer.lemmatize(i, pos='v') for i in lr_s])
    return ' '.join(l)

def stop_word(content):
    words = lem(content)
    op = [i for i in words.split(" ") if not i in stop_words]
    return op

vocab = []
all_patterns = [] 

for intent in j_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        tokenized = stop_word(pattern)
        vocab.extend(tokenized)
        all_patterns.append((tokenized, tag))

pvocab = sorted(set(vocab))
print(f"Vocabulary Size: {len(pvocab)}")  # Ensure it's 26
x = []
y = []

for (pattern_words, tag) in all_patterns:
    bag = [1 if w in pattern_words else 0 for w in pvocab]  # Use pvocab here
    x.append(bag)
    y.append(tag)

x = np.array(x)
y = np.array(y)

