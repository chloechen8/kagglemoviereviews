
# coding: utf-8

# In[26]:


# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


# In[27]:


import sentiment_read_subjectivity
#initialize the positive, neutral and negative word lists
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types(r'C:\Users\Xin\Downloads\CIS668\Term Project\FinalProjectData\FinalProjectData\kagglemoviereviews\SentimentLexicons\subjclueslen1HLTEMNLP05.tff')

#import sentiment_read_LIWC_pos_neg_words
# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
#(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

# define a feature definition function here

# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label
# import read_emotional_word


# In[40]:


def read_words():
    poslist = []
    neglist = []
    
    flexicon = open(r'C:\Users\Xin\Downloads\CIS668\Term Project\FinalProjectData\FinalProjectData\kagglemoviereviews\SentimentLexicons/NRC_emotion_lexicon_list.txt', encoding='latin1')
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


# In[41]:


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features


# In[42]:


def Emotion_features(document, word_features, poslist, neglist):
    document_word = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_word)
    posEmotion = 0
    otherEmotion = 0
    negEmotion = 0
    for word in document_word:
        if word in poslist:
            posEmotion += 1
        elif word in neglist:
            negEmotion += 1
        features['positivecount'] = 2 * posEmotion
        features['negativecount'] = 2 * negEmotion
    return features


# In[43]:


def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels

    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # computes evaluation measures for this fold and
        #   returns list of measures for each label
        print('Fold', i)
        (precision_list, recall_list, F1_list)                   = eval_measures(goldlist, predictedlist, labels)
        # take off triple string to print precision, recall and F1 for each fold
        '''
        print('\tPrecision\tRecall\t\tF1')
        # print measures for each label
        for i, lab in enumerate(labels):
            print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
              "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        '''
        # for each label add to the sums in the total lists
        for i in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[i] += precision_list[i]
            total_recall_list[i] += recall_list[i]
            total_F1_list[i] += F1_list[i]

    # find precision, recall and F measure averaged over all rounds for all labels
    # compute averages from the totals lists
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]
    # the evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),           "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels),           "{:10.3f}".format(sum(recall_list)/num_labels),           "{:10.3f}".format(sum(F1_list)/num_labels))

    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first intialize a dictionary for label counts and then count them
    label_counts = {}
    for lab in labels:
      label_counts[lab] = 0 
    # count the labels
    for (doc, lab) in featuresets:
      label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision),       "{:10.3f}".format(recall), "{:10.3f}".format(F1))


# In[44]:


def eval_measures(gold, predicted, labels):
    
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        if (TP == 0) or (FP == 0) or (FN == 0):
          recall_list.append (0)
          precision_list.append (0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)


# In[45]:


def vectorFeatureSet(document, word_features):
    tokens=document[0]
    vector = {}
    for word in word_features:
        vector[word]=False
        if word in tokens:
            vector[word]=True
    return wordVec


# In[46]:


# add NOT_features and POS tag
def NOT_features(document, word_features, negationwords):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
    	if tag.startswith('N'): numNoun += 1
    	if tag.startswith('V'): numVerb += 1
    	if tag.startswith('J'): numAdj += 1
    	if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features


# In[47]:


# One strategy with negation words is to negate the word following the negation word
#   other strategies negate all words up to the next punctuation
# Strategy is to go through the document words in order adding the word features,
#   but if the word follows a negation words, change the feature to negated word
# Start the feature set with all 2000 word features and 2000 Not word features set to false
def NNOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features


# In[48]:


# adding POS tag counts to the word features
def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


# In[49]:


#CSV file of features for our sample
def WekaFeature(word_features, document):
    f=open('./train.csv','r')
    count=0
    for phrase in word_features:
        f.write(str(count)+",")
        count+=1
        val= vectorFeatureSet(phrase,document).values()
        for i in val:
            if(i==False):
                f.write("0,")
            else:
                f.write("1,")
        f.write(str(phrase[1])+"\n")


# In[50]:


# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  for phrase in phraselist[:10]:
    print (phrase)
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  # add all the phrases
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
  
  # print a few
  for phrase in phrasedocs[:10]:
    print (phrase)

  # possibly filter tokens
  # lowercase - each phrase is a pair consisting of a token list and a label
  docs = []
  for phrase in phrasedocs:
    lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
    docs.append (lowerphrase)
  # print a few
  for phrase in docs[:10]:
    print (phrase)

  # continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  print(len(all_words))
  
  # get the 1500 most frequently appearing keywords in the corpus
  word_items = all_words.most_common(1500)
  word_features = [word for (word,count) in word_items]

  
  # feature sets from a feature definition function
  featuresets = [(document_features(d, word_features), c) for (d, c) in docs]
  
  # train classifier and show performance in cross-validation
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5
  cross_validation_PRF(num_folds, featuresets, labels)

  # NOT_featuresets 
  negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
  # define the feature sets
  NOT_featuresets = [(NNOT_features(d, word_features, negationwords), c) for (d, c) in docs]
  # show the values of a couple of example features
  cross_validation_PRF(num_folds, NOT_featuresets, labels)
  # f.close()
    
  """
  #POS tag feature
  POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in docs]
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5
  cross_validation_PRF(num_folds, POS_features, labels)
  """  
  

  # NOT_featuresets 
  negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
  # define the feature sets
  NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in docs]
  # show the values of a couple of example features
  cross_validation_PRF(num_folds, NOT_featuresets, labels)
  # f.close()
  
  POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in docs]
  cross_validation_PRF(num_folds, POS_featuresets, labels)
    
  (poslist, neglist) = read_words()  
  Emotion_featuresets = [(Emotion_features(d, word_features, poslist, neglist), c) for (d, c) in docs]
  cross_validation_PRF(num_folds, Emotion_featuresets, labels)


# In[51]:


processkaggle(r'C:\Users\Xin\Downloads\CIS668\Term Project\FinalProjectData\FinalProjectData\kagglemoviereviews\corpus',100)

