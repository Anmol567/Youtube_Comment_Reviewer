from flask import Flask,flash, render_template, url_for, request, redirect
import googleapiclient.discovery
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from gtts import gTTS
import heapq
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDp9zqixqm846mM_kH9LyNsUp95IMNMfiM"
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
import tensorflow_hub as hub
import re
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
import demoji
import re

from textblob import TextBlob
lemmatizer=WordNetLemmatizer()
word_to_id = imdb.get_word_index()
def videoidfetcher(link):
 url_data = urlparse.urlparse(link)
 query = urlparse.parse_qs(url_data.query)
 video = query["v"][0]
 return video
def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('sentiment_analysis.h5')
    graph = tf.compat.v1.get_default_graph()

youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = DEVELOPER_KEY)

stop=stopwords.words("english")
punc=[pun for pun in string.punctuation]
stop+=punc
print(stop)

import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def hello():
    return render_template('index.html')

def videoidfetcher(link):
 match = re.search("(?:\/|%3D|v=|vi=)([0-9A-z-_]{11})(?:[%#?&]|$)", link)
 if match:
    result = match.group(1)
 else:
    result = ""
 return result
def sent_anly_prediction(comment):
        words = comment.split()
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=5000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=500)
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            probability = model.predict(array([vector][0]))[0][0]
            print(probability)
            class1 = model.predict_classes(array([vector][0]))[0][0]
        if class1 == 0:
            return "Negative"
        else:
            return "Positive"

@app.route('/text', methods = ['POST'])
def predict_text():
   if request.method == 'POST':
    text = request.form['text']
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    comment = text.lower().replace("<br />", " ")
    comment=re.sub(strip_special_chars, "", comment.lower())
    g=TextBlob(comment).sentiment.polarity
    g=int((g+1)*50)
    result_dic = {
    'positive':g,
    'negative':100-g,
    'text':text,
    }
    print(g)
    return render_template('index.html',prediction=result_dic)
@app.route('/', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
     my_colors={}
     my_colors[1]="primary"
     my_colors[2]="secondary"
     my_colors[3]="success"
     my_colors[4]="danger"
     my_colors[5]="warning"
     
     text = request.form['youtube_video_url']
     video_id= videoidfetcher(text)
     if(video_id==""):
      flash('Looks like you have entered invalid youtube link!!!')
      return render_template('index.html')
     print(video_id)
      
     heap_of_good_likes=[]
     most_liked_comments=[]
     query_results =youtube.commentThreads().list(part="snippet",maxResults=100,textFormat="plainText",order="relevance",videoId=video_id).execute()
     negative=0
     positive=0
     for x in (query_results['items']):
      comment=x['snippet']['topLevelComment']['snippet']['textDisplay']
      strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
      comment = comment.lower().replace("<br />", " ")
      comment=re.sub(strip_special_chars, "", comment.lower())
      cleaned_comment=comment
      if(TextBlob(cleaned_comment).sentiment.polarity<0):
       print(cleaned_comment)
       print(TextBlob(cleaned_comment).sentiment.polarity)
       negative=negative+1
      else:
       print(cleaned_comment)
       print(TextBlob(cleaned_comment).sentiment.polarity)
       positive=positive+1
       
      get_like_count=x['snippet']['topLevelComment']['snippet']['likeCount']
      if len(heap_of_good_likes)<5:
       heapq.heappush(heap_of_good_likes,(get_like_count,comment));
      else:
       top=heapq.heappop(heap_of_good_likes)
       if(top[0]<get_like_count):
        heapq.heappush(heap_of_good_likes,(get_like_count,comment));
       else:
        heapq.heappush(heap_of_good_likes,top)
    while heap_of_good_likes:
     most_liked_comments.append(heapq.heappop(heap_of_good_likes))
    most_liked_comments.reverse()
    my_positive=int((positive/(positive+negative))*100)
    my_negative=100-my_positive
    result_dic = {
      'positive':my_positive,
      'negative':my_negative,
      'youtube_video':video_id,
      'most_liked_comments':most_liked_comments,
      'mycolors':my_colors
      }
    return render_template('index.html',results=result_dic)

def get_simple_POS(tag):
    if tag.startswith('J'):
      return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def cleanwords(sentence):
   sentence_emogis=demoji.findall(sentence)
   sentence_emogis_short=" "
   for value in sentence_emogis.values():
    sentence_emogis_short=sentence_emogis_short+(str(value)+" ")
   sentence=deEmojify(sentence)
   words=word_tokenize(sentence)
   words=[lemmatizer.lemmatize(word,pos=get_simple_POS(pos_tag(word)[0][1])).lower() for word in words if not word.lower() in stop and not word.isdigit()]
   return " ".join(words)
  

if __name__ == '__main__':
    init()
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.run(debug = False)
