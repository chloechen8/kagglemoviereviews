
# coding: utf-8

# In[11]:


import os
import sys


# In[79]:


def read_words():
    poslist = []
    neglist = []
    
    flexicon = open('SentimentLexicons/NRC_emotion_lexicon_list.txt', encoding='latin1')
    wordlines = [line.strip() for line in flexicon]
    for line in wordlines:
        if not line == '':
            items = line.split()
            for c in items:
                if items[1] == 'positive' and items[2] == '1':
                    poslist.append(items[0])
                elif items[1] == 'negative' and items[2] == '1':
                    neglist.append(items[0])
            
    return(poslist,neglist)


# In[84]:


if __name__=='__main__':
  (poslist, neglist) = read_words()
  print ("Positive Words", len(poslist), "Negative Words", len(neglist))
  print (poslist)
  print (neglist)

