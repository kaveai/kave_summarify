from flask import Flask, render_template, make_response, request, redirect,send_file,Response
from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import requests 
import datetime
#import sqlite3
from nltk.corpus import stopwords
import heapq
from gensim.summarization import keywords
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import json
import pickle
from keras.models import model_from_json
from keras.models import load_model
from gensim.models import KeyedVectors
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Doc2Vec
import tensorflow as tf
from time import sleep
app = Flask(__name__)

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/uploads/')

###--------FONKSİYONLAR------###
def GetText(url):
    html = requests.get("http://www.hurriyet.com.tr/"+url).text
    soup = bs(html, "lxml")
    try:
        body = soup.find("div", class_="rhd-all-article-detail").findAll('p')
    except AttributeError:
        try:
            body_text = soup.findAll("div", class_="news-box")[1].find('p').text
            summarized_text=soup.findAll("div", class_="news-detail-spot news-detail-spot-margin")[0].find('h2').text
            header = soup.find("h2", class_="news-detail-title selectionShareable local-news-title").text
            time = soup.find("div", class_="col-md-8 text-right").text[:10]
            return (body_text,summarized_text,header,time)
        except:
            header = soup.find("h1", class_="rhd-article-title").text
            time = soup.find("div", class_="rhd-time-box").text[:10]
            body = soup.findAll("h3", class_="description")
            body_text = ''
            for element in body:
                body_text += ''.join(element.findAll(text = True))
            summarized_text = soup.find("h2", class_="rhd-article-spot").text
            return (body_text,summarized_text,header,time)
    body_text = ''
    for element in body:
            body_text += ''.join(element.findAll(text = True))
    # Koruma
    if len(body_text) == 0:
        body_text = soup.find("div", class_="rhd-all-article-detail").text
        summarized_text = soup.find("h2", class_="rhd-article-spot").text
    try:
        header = soup.find("h2", class_="news-detail-title selectionShareable local-news-title").text
        time = soup.find("div", class_="col-md-8 text-right").text[:10]
    except AttributeError:
        header = soup.find("h1", class_="rhd-article-title").text
        time = soup.find("div", class_="rhd-time-box").text[:10]
        summarized_text = soup.find("h2", class_="rhd-article-spot").text
        return (body_text,summarized_text,header,time)
    return (body_text,summarized_text,header,time)

class extraction_based_sum():
    def __init__(self):
        self.jstr = json.loads(open(UPLOADS_PATH+'kokbulma.json').read())
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.model = model_from_json(self.jstr)
                self.model.load_weights(UPLOADS_PATH+'model.hdf5')

        self.word_tr = KeyedVectors.load_word2vec_format(UPLOADS_PATH+'trmodel.dms', binary=True)
        fp = open(UPLOADS_PATH+'datafile.pkl','rb')
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

        with self.graph.as_default():
            with self.session.as_default():
                yp = self.model.predict(X)
                return self.decode(yp[0])
    
    def cleanText(self,text):
        
        text_file = open(UPLOADS_PATH+"turkce-stop-words.txt", "r")
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
        ranked_sentences = sorted((s for i,s in enumerate(sentences)), reverse=True)
        return ranked_sentences
    
    def get_sentences(self,text,sum_length):
        ranked_sentences = self.cleanText(text)
        #print(ranked_sentences)
        summary = []
        for i in range(sum_length):
            #print(ranked_sentences[i],i)
            summary.append(ranked_sentences[i])
        text = " ".join(summary)
        return text
        
    
    def get_keywords(self,text,ratio):
        text_keywords = keywords(text,ratio=ratio).split("\n")
        valid_keywords = []
        for keyword in text_keywords:
            if keyword not in self.stop_words:
                valid_keywords.append(keyword)
        return valid_keywords

# Haber benzerliği için class'ımız

class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = str(doc).lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Kelime sözlükte yoksa yok sayıyoruz.
                pass

        # Döküman vektörleri tüm kelime vektörlerinin ortalaması olduğunu kabul ediyoruz.
        # Not: Daha iyi yollar da var
        vector = np.mean(word_vecs, axis=0)
        return vector


    def _cosine_sim(self, vecA, vecB):
        """İki vektör arasındaki cosinüs benzerliğini hesaplıyoruz"""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        """Verilen haberle diğer haberleri karşılaştırarak benzerlik skorlarını hesaplıyıp
            sonuç olarak döndürüyoruz."""
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        for doc in target_docs:
            target_vec = self.vectorize(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({
                    'score' : sim_score,
                    'doc' : doc
                })
            results.sort(key=lambda k : k['score'] , reverse=True)

        return results
###--------FONKSİYONLAR------###
ex_sum = extraction_based_sum()

model_path = UPLOADS_PATH+'trmodel.dms'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
file=open(UPLOADS_PATH+'turkce-stop-words.txt', 'r')
stopwords = list(file.read().split())

ds = DocSim(w2v_model,stopwords)

@app.route('/')
def index()->'html':
    return render_template('index.html')


@app.route('/demo')
def demo_base()->'html':
    return render_template('demo.html')


@app.route('/demo_sonuc', methods=['POST'])
def demo_sonuc()->'html':
    link = request.form['articlelink']
    link = link.split(".com.tr")[1]
    body_text, sum_text, header, time = GetText(link)
    try:
        new_sum_text = ex_sum.get_sentences(body_text,2)
    except IndexError:
        new_sum_text = body_text
    except ValueError:
        new_sum_text = body_text
    return render_template('demo_sonuc.html',body_text=body_text, new_sum_text=new_sum_text, header=header, time=time, link=link)

@app.route('/gundem')
def gundem()->'html':
    url = 'http://www.hurriyet.com.tr/index/?d=20191201'
    html = requests.get(url).text
    soup = bs(html, "lxml")

    urlx = []
    for links in soup.findAll("div", class_="news"):
        urlx.append(links.find('a').get('href'))

    url = urlx[10:20]
        
    for c in url:
        if c[:6] == "/video":
            url.remove(c)
            url.append(urlx[np.random.randint(20,len(urlx))])
        else:
            continue
        
    data = []
    for i in url:
        data.append(GetText(i))

    data = pd.DataFrame(data)
    data.columns = ["body_text","summarized_text","header","time"]
    sum_texts = []
    for sums in data.body_text.values:
        try:
            sum_texts.append(ex_sum.get_sentences(sums,2))
        except IndexError:
            sum_texts.append(sums)
            continue
        except ValueError:
            sum_texts.append(sums)
            continue
    headers = data.header.values
    times = data.time.values
    urls = url
    return render_template('gundem.html', sum_texts=sum_texts, headers=headers, times=times, urls=url)

@app.route('/select_article')
def select_article()->'html':
    data = pd.read_csv(UPLOADS_PATH+'Sample.csv')
    lucky = data.loc[np.random.randint(0,len(data))]
    times = lucky.time
    sum_texts = lucky.body_text
    headers = lucky.header
    return render_template('select_article.html',times=times,sum_texts=sum_texts, headers=headers)

@app.route('/liked/<text>')
def make_reco(text):
    data=pd.read_csv(UPLOADS_PATH+'7000_haber.csv') 
    data.drop(columns=['Unnamed: 0'],inplace=True)

    source_doc= text#Karşılaştırmak istediğimiz haber
    target_docs=list(data['Cleaned']) #Karşılaştırıldığı diğer haberler

    sim_scores = ds.calculate_similarity(source_doc, target_docs) #Benzerliğin hesaplanması

    tarihler,basliklar,body_text = [],[],[]
    for docs in sim_scores[1:11]:
        spec = data[data.Cleaned == docs['doc']]
        tarihler.append(spec.Tarıh.values[0])
        basliklar.append(spec.Baslık.values[0])
        body_text.append(spec.Cleaned.values[0])

    sum_texts = []
    for sums in body_text:
        try:
            sum_texts.append(ex_sum.get_sentences(sums,2))
        except IndexError:
            sum_texts.append(sums)
            continue
        except ValueError:
            sum_texts.append(sums)
            continue
    return render_template('sana_ozel.html', tarihler=tarihler, basliklar=basliklar, sum_texts=sum_texts)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=35000)