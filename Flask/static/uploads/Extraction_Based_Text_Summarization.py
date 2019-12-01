#!/usr/bin/env python
# coding: utf-8

# # Index
# * [Necessary Libraries](#Necessary-Libraries)
# * [Extraction Based Sum as Class](#Extraction-Based-Sum-as-Class)
# * [Usage Example](#Usage-Example)

# # Necessary Libraries

# In[6]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import heapq
from gensim.summarization import keywords
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import numpy as np
import json
import pickle
from keras.models import model_from_json
from keras.models import load_model


# # Extraction Based Sum as Class

# In[45]:


class extraction_based_sum():
    def __init__(self):
        self.jstr = json.loads(open('kokbulma.json').read())
        self.model = model_from_json(self.jstr)
        self.model.load_weights('model.hdf5')
        self.word_tr = KeyedVectors.load_word2vec_format('trmodel.dms', binary=True)
        fp = open('datafile.pkl','rb')
        data = pickle.load(fp)
        fp.close()
        self.chars = data['chars']
        self.charlen = data['charlen']
        self.maxlen = data['maxlen']
        
    def encode(self,word,maxlen=22,is_pad_pre=False):
        wlen = len(str(word))
        if wlen > maxlen:
            word = word[:maxlen]

        word = str(word).lower()
        pad = maxlen - len(word)
        if is_pad_pre :
            word = pad*' '+word   
        else:
            word = word + pad*' '
        mat = []
        for w in word:
            vec = np.zeros((self.charlen))
            if w in self.chars:
                ix = self.chars.index(w)
                vec[ix] = 1
            mat.append(vec)
        return np.array(mat)

    def decode(self,mat):
        word = ""
        for i in range(mat.shape[0]):
            word += self.chars[np.argmax(mat[i,:])]
        return word.strip()
    
    def kokBul(self,word):
        X = []

        w = self.encode(word)
        X.append(w)

        X = np.array(X)

        yp = self.model.predict(X)
        return self.decode(yp[0])
    
    def cleanText(self,text):
        
        text_file = open("turkce-stop-words.txt", "r")
        lines = text_file.readlines()
        self.stop_words = []
        for line in lines:
            self.stop_words.append(line[:-1])
        self.stop_words.append('bir')
        self.stop_words.append('bin')
        text = re.sub(r'[\s]',' ',text)
        sentences = sent_tokenize(text)
        self.clean_sentences = []
        for sentence in sentences:
            temp_list = []
            for word in sentence.split():
                if (word.lower() not in self.stop_words) and (len(word) >= 2):
                    temp_list.append(self.kokBul(word))
            self.clean_sentences.append(' '.join(temp_list))
        sentence_vectors = []
        for sentence in self.clean_sentences:
            for word in sentence.split():
                try:
                    v = word_tr[word.lower()]
                except:
                    v = np.zeros(400)
                sentence_vectors.append(v)
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,400), sentence_vectors[j].reshape(1,400))[0,0]
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        return ranked_sentences
    
    def get_sentences(self,text,sum_length):
        ranked_sentences = self.cleanText(text)
        summary = []
        for i in range(sum_length):
            summary.append(ranked_sentences[i][1])
        return " ".join(summary)
        
    
    def get_keywords(self,text,ratio):
        text_keywords = keywords(text,ratio=ratio).split("\n")
        valid_keywords = []
        for keyword in text_keywords:
            if keyword not in self.stop_words:
                valid_keywords.append(keyword)
        return valid_keywords


# # Usage Example

# In[50]:


ex_sum = extraction_based_sum()


# In[47]:


text = """
Transition-One adlı girişim, donanım iyileştirme teknolojisiyle eski dizel araçları elektrikli araca dönüştürüyor.

Fransız girişimi Transition-One, eski dizel araçlara 8 bin 500 Euro karşılığında elektrik motoru, batarya ve bağlantılı bir gösterge paneli ekleyen donanım iyileştirme teknolojisi geliştirdi.

Transition-One kurucusu Aymeric Libeau “Yeni bir elektrikli arabaya 20 bin Euro veremeyecek durumdaki insanlara ulaşmayı hedefliyorum.” diyor. 2009 model bir Renault Twingo’yu 180 kilometre menzilli bir elektrikli araca dönüştürdüğü ilk prototipini gösteren Libeau “Avrupa’da en çok satılan modelleri elektrikli arabalara dönüştürüyoruz.” dedi.

Dönüşüm bir günden az sürüyor.

Libeau, bu yılın sonuna kadar Fransız ve Avrupalı düzenleyicilerden onay almayı umuyor. Ayrıca talep durumunu test etmek için Eylül ayında ön sipariş almaya başlayacak. Otomobil üreticileri, Avrupa’daki katı karbon salınımı düzenlemelerine uyabilmek için hızla elektrikli araba üretmeye çalışıyor. Eski dizel arabaları yasaklayan şehirlerin sayısı her geçen gün artıyor. Önümüzdeki on yıl içinde de çok daha fazla Avrupa şehri fosil yakıtlı arabalara erişimi kesecek.

Libeau’nun yöntemiyle dizel aracı elektrikliye dönüştürme işlemi bir günden az sürüyor.
"""


# In[52]:


ex_sum.get_sentences(text,5)


# In[49]:


ex_sum_5.get_keywords(text,0.25)


# In[ ]:




