import os
import pytesseract
from cv2 import cv2
try:
 from PIL import Image
except ImportError:
 import Image
import base64
import argparse
from pdf2image import convert_from_path
import urllib3
import urllib.request
from urllib.request import urlretrieve, Request
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from collections import defaultdict as dc
nltk.download('punkt')
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from collections import defaultdict as dc
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import requests
import io
import textract
import warnings
# import cloudinary as Cloud
warnings.filterwarnings("ignore")
directory = os.getcwd()

# Cloud.config.update = ({
#     'cloud_name':'read-it',
#     'api_key': '972376323111169',
#     'api_secret': 'nS4CyxoR-7BPV8uSG8EIdEEn71c'
# })

class Text:
  def __init__(self,no_of_topics=10, scan=False):
    self.file_path=None
    self.text=None
    self.url=None
    self.no_of_topics=no_of_topics
    self.scan=scan
    #self.no_of_sent=None
    #self.positive_count
    stopword_=pd.read_pickle(directory+'\\Text\\dictionary\\text_stopwords.pkl')
    positive_dict=pd.read_pickle(directory+'\\Text\\dictionary\\text_pos.pkl')
    negative_dict=pd.read_pickle(directory+'\\Text\\dictionary\\text_neg.pkl')
    constraints=pd.read_pickle(directory+'\\Text\\dictionary\\text_constraints.pkl')
    uncertain=pd.read_pickle(directory+'\\Text\\dictionary\\text_uncertain.pkl')
    
    # stopword_=pd.read_pickle('C:\\Users\\vishu\\OneDrive\\Desktop\\Vishal\\final year project\\Text-analysis~\\Final_project_Server-main\\Text\\dictionary\\text_stopwords.pkl')
    # positive_dict=pd.read_pickle('C:\\Users\\vishu\\OneDrive\\Desktop\\Vishal\\final year project\\Text-analysis~\\Final_project_Server-main\\Text\\dictionary\\text_pos.pkl')
    # negative_dict=pd.read_pickle('C:\\Users\\vishu\\OneDrive\\Desktop\\Vishal\\final year project\\Text-analysis~\\Final_project_Server-main\\Text\\dictionary\\text_neg.pkl')
    # constraints=pd.read_pickle('C:\\Users\\vishu\\OneDrive\\Desktop\\Vishal\\final year project\\Text-analysis~\\Final_project_Server-main\\Text\\dictionary\\text_constraints.pkl')
    # uncertain=pd.read_pickle('C:\\Users\\vishu\\OneDrive\\Desktop\\Vishal\\final year project\\Text-analysis~\\Final_project_Server-main\\Text\\dictionary\\text_uncertain.pkl')
    self.stopwords=dc(int)
    self.positive_dict_13=dc(int)
    self.negative_dict_13=dc(int)
    self.constraints_dict=dc(int)
    self.uncertain_dict=dc(int)
    for i in stopword_:
      self.stopwords[i]=1
    for i in positive_dict:
      self.positive_dict_13[i]=1
    for i in negative_dict:
      self.negative_dict_13[i]=1
    for i in constraints:
      self.constraints_dict[i]=1
    for i in uncertain:
      self.uncertain_dict[i]=1

  def __pdf_to_image(self,file_path):
      # using convert_from_path method to access pdf by passing its dstination
      #path and converting into image list.
    pages = convert_from_path(file_path, poppler_path=directory+"\\Text\\poppler-0.68.0\\bin")
    return pages

  def __image_preprocessing(self,im):
  #im-->PIL image obtained from pdf
  #save image as jpeg file in this notebook
    im.save(directory+'\\Text\\TextD\\image.jpg', 'JPEG')
    '''
    Reading the saved image using opencv imread function.
    imread reads images in BGR format so to convert it into RGB,
    by using cv2.COLOR_BGR2RGB parameter.
    Image is stored in variable img
    '''
    img=cv2.imread(directory+'\\Text\\TextD\\image.jpg', cv2.COLOR_BGR2RGB )
    #print(img.shape)
    # converting the img into grayscale mode, i.e., reducing from 3 channels to 2
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #plt.imshow(img)
    #plt.show()
    '''
    Using Otsu binarization threshold to automatically
    choose a threshold for thresholding the image such that
    all pixel valus below it are 0 and rest to the maximim
    '''
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
  #print(img.shape)    
  #cv2.imwrite(r"./preprocess/img_threshold.png",img)
  #plt.imshow(img)
  #plt.show()
    '''
    Morphological Filters are applied to remove noise and 
    smoothen the image.
    A struturing element of dimension (4,4) is chosen fro applying the
    morphological transfromations
    '''
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    #print(img.shape)
    '''
    considering tophat and balckhat morphological filter
    to join broken parts and smoothening the edge
    '''
    tophat=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    #blackhat=cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    #print(img.shape)
    add=cv2.add(img,tophat)
    #sub=cv2.subtract(add,blackhat)
    #print(add.shape)
    #print(sub.shape)
    #t=threshold_local(sub,29,  offset=35, method="gaussian", mode="mirror")
    #thresh=(sub>t).astype("uint")*255
    #thresh_=cv2.bitwise_not(thresh)
    #print(thresh_.shape)
    #thresh_=np.moveaxis(thresh_, 0, 2)
    #plt.imshow(add)
    #plt.show()
    if add.shape!=(2200,1700):
      add=cv2.resize(add, (1700,2200), interpolation=cv2.INTER_AREA)
    #plt.imshow(add)
    #plt.show()
    return add

  
  def __extract_text_img(self,im):
    return pytesseract.image_to_string(im)
        

  def file_path_ext(self,file_path=None, url=None,text=None):
    self.file_path=file_path
    self.text=text
    self.url=url
    #print(file_path,text,url)
    
    if self.scan==True:
      pg=self.__pdf_to_image(file_path)
      text=""
      for i in pg:
        img=self.__image_preprocessing(i)
        if text=="":
          text=self.__extract_text_img(img)
        else:
          text=text+" "+self.__extract_text_img(img)
    elif url:
      if(url.startswith("http") or url.startswith("www.")) and (url.endswith(".pdf") or url.endswith(".txt") or url.endswith(".docx") or url.endswith(".odt") or url.endswith(".doc")):
        s=url.split('/')
        print(s)
        print(url)
        
        #req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        #print(req)
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', 'whatever')
        filename, headers = opener.retrieve(url, directory+"\\Text\\TextD"+str(s[-1]))
        #urlretrieve(url,directory+"\\"+str(s[-1]))
        text=textract.process(directory+"\\Text\\TextD"+str(s[-1]))
        text=text.decode('unicode_escape').encode('ascii', 'ignore')
        text=text.decode("utf-8")
        #text="vishal rochlani hsdjksh hajshfk hdjshk"
      else:
        text=self.__text_extract_url(url)
    elif file_path:
      text = textract.process(file_path)
      #text = text.decode("utf-8")
      text=text.decode('unicode_escape').encode('ascii', 'ignore')
      text=text.decode("utf-8")
      #text=self.__extract_text_pdf_txt(self.file_path)
    text=text.replace("\n"," ")
    text=text.replace("\r"," ")
    text=text.replace("\t"," ")
    text=text.replace("\x0c"," ")
    text=" ".join(text.split(" "))
    
    text=text.replace("\n"," ")
    text=" ".join(text.split(" "))
    self.text=text
    


  #for readability
  def __text_extract_url(self,file_path):

    url=file_path
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data)
    # get text
    for script in soup(["script", "style"]):
      script.extract()
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    #print(text)
    return text

  def __preprocess_text_readability(self,text):
    sent=sent_tokenize(text)
    #print(sent)
    no_of_sent=len(sent)
    word=[]
    #tokenizing words from each of the sentence using regexp tokenizer stored as tokenize while importing the package
    for s in sent:
      word.extend(tokenizer.tokenize(s))
    #print(len(word))
    preprocess=[]
    for s in word:
      # converting into lowercase
      s=s.lower()
      if self.stopwords[s]==0:
        #lemmatizing the filtered words.
        preprocess.append(wordnet_lemmatizer.lemmatize(s))
      #returns the preprocessed tokenize text and total no of sentences
    #print(len(preprocess))
    for i in range(len(preprocess)):
      preprocess[i]=preprocess[i].replace('\n'," ")
      preprocess[i]=''.join(preprocess[i].split())
    #print(preprocess)
    return [preprocess, no_of_sent]

  def __positive_score(self,sec):
    positive_count=0
    for i in sec:
      if self.positive_dict_13[i]==1:
        positive_count+=1
    #returns positive score
    return positive_count

  #function for computing the negattive score of any section
  def __negative_score(self,sec):  
    negative_count=0
    for i in sec:
      if self.negative_dict_13[i]:
        negative_count+=1
  #returns negattive score
    return negative_count

  def __polarity(self,ps,ns):
    return (ps - ns)/((ps + ns) + 0.000001)

  def __avg_sent_len(self,nw,ns):
    return nw/ns

  def __syllable_count(self,s):
    vowel="aeiouy"
    count=0
    # for counting vowels
    if s[0] in vowel:
      count+=1
    for i in range(1,len(s)):
      #for dipthongs and tripthongs
      if s[i] in vowel and s[i-1] not in vowel:
        count+=1
      #for removing silent sounds
    if s.endswith('e') or s.endswith("ed") or s.endswith("es"):
      count-=1
    if s.endswith("le") and len(s)>=3 and s[-3] not in vowel:
      count+=1
    if count==0:
      count=1
    return count

  def __complex_word(self,sec):
    no_of_complex=0
    for s in sec:
      if self.__syllable_count(s)>2:
        no_of_complex+=1
    return no_of_complex

  def __percentage_complex(self,nc,nw):
    return nc/nw

  def __fog_index(self,asl,pc):
    return 0.4*(asl+pc)

  def __constraining_score(self,sec):
    constraints_count=0
    for s in sec:
      if self.constraints_dict[s]:
        constraints_count+=1
    return constraints_count

  def __uncertainity_score(self,sec):  
    uncertain_count=0
    for s in sec:
      if self.uncertain_dict[s]:
        uncertain_count+=1
    return uncertain_count

  def __proportion(self,ps,ns,cs,us,nw):
    return [ps/nw, ns/nw, cs/nw, us/nw]

  #return: dictionary of readability scores
  def readability_analysis(self):

    sec,no_of_sent=self.__preprocess_text_readability(self.text)
    #print(sec,no_of_sent)
    #return text_preprocess, self.no_of_sent
    #sec_p=[]
    no_of_words=len(sec)
    positive_score_=self.__positive_score(sec)
    negative_score_=self.__negative_score(sec)
    polarity_=self.__polarity(positive_score_, negative_score_)
    avg_sent_len_=self.__avg_sent_len(no_of_words, no_of_sent)
    no_of_complex_word=self.__complex_word(sec)
    complex_percentage=self.__percentage_complex(no_of_complex_word,no_of_words)
    fog_index_=self.__fog_index(avg_sent_len_,complex_percentage)
    constraining_score_=self.__constraining_score(sec)
    uncertainity_score_=self.__uncertainity_score(sec)
    positive_proportion, negative_proportion,constraining_proportion, uncertainity_proportion=self.__proportion(positive_score_, negative_score_, constraining_score_, uncertainity_score_, no_of_words)
    return {"positive score":positive_score_,"negative score":negative_score_,"polarity_":polarity_,"complex_percentage":complex_percentage,"fog_index_":fog_index_,"constraining_score_":constraining_score_,"uncertainity_score_":uncertainity_score_,"positive_proportion":positive_proportion, "negative_proportion":negative_proportion,"constraining_proportion":constraining_proportion, "uncertainity_proportion":uncertainity_proportion}

  def __preprocess_text_pylda(self,sec):
    sent=sent_tokenize(sec)
    #print(sent)
    no_of_sent=len(sent)
    word=[]
    #tokenizing words from each of the sentence using regexp tokenizer stored as tokenize while importing the package
    for s in sent:
      s_word=tokenizer.tokenize(s)
      p=[]
      for w in s_word:
        w=w.lower()
        if self.stopwords[w]==0:
          p.append(w)
      word.append(p)
    return word

  '''
  return: 1. dictionary of topics with words and their percent contri
  # 2. dictionary with key as topic no and value a slist of words in that topic
  3. text extracted
  '''
  def upload_cloudinary(self,path):
    pipeshelf = Cloud.CloudinaryImage(path)
    return pipeshelf.url

  def topic_modelling(self):
    sec=self.__preprocess_text_pylda(self.text)
    dictionary = corpora.Dictionary(sec)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in (sec)]
    LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=self.no_of_topics, random_state=100,
                chunksize=1000, passes=50)
    #pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis,directory+"\\templates\\abc.html")
    #Clodinary_url = self.upload_cloudinary(directory+"\\Text\\Output\\abc.html")
    pyLDAvis.save_json(vis,directory+"\\Text\\Output\\abc.json")
    p=list(lda_model.print_topics())
    topics_=dict()
    for i in p:
      topics_[i[0]]=i[1]
    topic_list=dict()
    for i in topics_:
      s=topics_[i]
      s=s.split("+")
      #Sprint(s)
      t=[]
      for i in s:
        q=i.split("*")
        #print(q[1][1:-2])
        t.append(q[1][1:-2])
      topic_list[i]=t
      #print(t)
      #p=[s[j] for j in range(1,len(s),2)]
      #print(p)
      #for j in range(len(s)):
        #s[j]
    Clodinary_url="http://localhost:5000/abc.html"
    return topics_, topic_list, self.text,Clodinary_url
      
    

  #return summary as string
  def extractive_summary(self):
    try:
      return summarize(self.text,ratio=0.5)
    except:
      return None