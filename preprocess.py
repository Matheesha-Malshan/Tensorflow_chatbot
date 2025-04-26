import nltk
from jread import j_data
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import re


lemmatizer=WordNetLemmatizer()

def lower_casing(content):
    return content.lower()


def remove_punctuation(content):
    return re.sub(r'[^\w\s]', '',lower_casing(content))


def lem(content):
    
    lr=remove_punctuation(content)
    lr_s=lr.split(" ")
    l= ([lemmatizer.lemmatize(i,pos='v') for i in lr_s])
    return ' '.join(l)

def stop_word(content):
    words=lem(content)
    stop_words=set(stopwords.words('english'))
    
    op=[i for i in words.split(" ") if not i in stop_words]
    return ' '.join(op).split(" ")

def vectors(content,vec_len,j_data):
    c_p=stop_word(content)
    len_c=len(c_p)
    data_=np.ones([vec_len,len_c])
    h=[]
    for i in j_data['intents']:
        p = i["patterns"]
        for k in p:
          #  h.append("".join(k))
          print(k)
"""
    for i,j in enumerate(h):
        for l in h[i]:
            if not l in c_p:
                data_[]
"""
    

vec_len=0
paragraph=" "
for i in j_data['intents']:
    p = i["patterns"]
    vec_len+=len(p)
    paragraph += " ".join(p) + " "  

#print(stop_word(paragraph,vec_len))
#print(vec_len)

print(vectors(paragraph,vec_len,j_data))