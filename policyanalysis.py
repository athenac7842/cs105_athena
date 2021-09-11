#!/usr/bin/env python
# coding: utf-8

# Analyze Privacy Policies

# https://pypi.org/project/textstat/
# https://colab.research.google.com/github/mohammedterry/NLP_for_ML/blob/master/Sentiment_Analysis.ipynb#scrollTo=0PWxpQSAO6x2
# https://medium.com/@prakash507979/how-to-read-pdf-file-using-python-1e4269a5f75f
# https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/4-files/PY0101EN-4-1-ReadFile.ipynb

# In[ ]:





# In[1]:


#packages (do once)
get_ipython().system('pip install textstat')


# In[40]:


#libraries
import textstat
import pandas as pd
import os
import glob

from collections import Counter

#sentiment analysis
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[81]:


#count word frequencies
def wordfreq(text):
  ltext = text.lower()
  skips = [".", ",",":", ";", "!", "?",'"']
  for i in skips: 
    ltext = ltext.replace(i, "")

  wordfreq = Counter(ltext.split(" "))
  return wordfreq

def word_stats(wordfreq):
  num_unique = len(wordfreq(ppolicy))
  counts = wordfreq(ppolicy).values()
  ratio = num_unique/sum(counts)
  bullets = ppolicy.count('*')
  complexity = textstat.flesch_reading_ease(ppolicy)
  readability = textstat.text_standard(ppolicy, float_output=False)
  rtimemin = (sum(counts)/250)
  sentiment = sid.polarity_scores(ppolicy)
  
  return ratio
  #return (num_unique, sum(counts), bullets, complexity, readability, rtimemin, sentiment)


# In[82]:


#load files
stats_list = []

files = glob.glob('/Users/Athena/Desktop/privacypolicies/*.txt')

for file in files:
    with open(file, "r", encoding='utf8', errors = 'ignore') as f:
        ppolicy = f.read()
        stats = word_stats(wordfreq)
        stats_list.append(stats)
        stats_list.append("___")
        print(file)

print(stats_list)
        
    


# In[59]:


text = "According to the Duolingo privacy policy, shares non-personal data with third party providers which can be linked back to your personal information. Privacy"


# In[60]:


word_stats(wordfreq)


# In[ ]:




