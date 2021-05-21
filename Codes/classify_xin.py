# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.corpus import sentence_polarity
import random
import re
import sentiment_read_subjectivity
# initialize the positive, neutral and negative word lists
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('SentimentLexicons/subjclueslen1HLTEMNLP05.tff')

import sentiment_read_LIWC_pos_neg_words
# initialize positve and negative word prefix lists from LIWC
#   note there is another function isPresent to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

# function that takes a word and returns true if it consists only
#   of non-alphabetic characters  (assumes import re)
def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False


# define a feature definition function here

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features


# create feature extraction function that has all the word features as before, but also has bigram features.
def bigram_document_features(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    return features

# adding POS tag counts to the word features
def POS_features(document):
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


def confusion_matrix():
    goldlist = []
    predictedlist = []
    for (features, label) in test_set:
    	goldlist.append(label)
    	predictedlist.append(classifier.classify(features))
    print("The confusion matrix is ")
    cm = nltk.ConfusionMatrix(goldlist, predictedlist)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# cross-validation:
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
        (precision_list, recall_list, F1_list) \
                  = eval_measures(goldlist, predictedlist, labels)
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
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels), \
          "{:10.3f}".format(sum(recall_list)/num_labels), \
          "{:10.3f}".format(sum(F1_list)/num_labels))

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
    print( '\t', "{:10.3f}".format(precision), \
      "{:10.3f}".format(recall), "{:10.3f}".format(F1))


# define features that include word counts of subjectivity words
# negative feature will have number of weakly negative words +
#    2 * number of strongly negative words
# positive feature has similar definition
#    not counting neutral words
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features


# One strategy with negation words is to negate the word following the negation word
#   other strategies negate all words up to the next punctuation
# Strategy is to go through the document words in order adding the word features,
#   but if the word follows a negation words, change the feature to negated word
# Start the feature set with all 2000 word features and 2000 Not word features set to false
def NOT_features(document, word_features, negationwords):
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


# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output:  prints precision, recall and F1 for each label

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
  ###################################################################################
  # each phrase has a list of tokens and the sentiment label (from 0 to 4)
  ### bin to only 3 categories for better performance
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))

  # possibly filter tokens
  # lowercase - each phrase is a pair consisting of a token list and a label
  docs = []
  for phrase in phrasedocs:
    lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
    docs.append (lowerphrase)
  # print a few
  for phrase in docs[:10]:
    print (phrase)

  # possibly filter tokens
  stopwords = nltk.corpus.stopwords.words('english')
  #remove stopwords
  for phrase in docs:
      alphaTokens = [w for w in docs if not alpha_filter(w)]
      stopTokens = [w for w in alphaTokens if not stopwords]

  # continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  print(len(all_words))


  bag_of_word = []
  # continue as usual to get all words and create word features
  for tokens in stopTokens:
      t = tokens[0]
      for t in tokens:
          if t not in bag_of_word:
              bag_of_word.append(t)

  # get the 1500 most frequently appearing keywords in the corpus
  # limit the length of word features to 500
  all_words = nltk.FreqDist(bag_of_word)
  word_items = all_words.most_common(1500)
  word_features = [word for (word, count) in word_items]

  # feature sets from a feature definition function
  featuresets = [(document_features(d, word_features), c) for (d, c) in docs]
 # feature sets from a feature definition function
  print("featuresets*********************")
  len(featuresets)

  # train classifier and show performance in cross-validation
  # make a list of labels
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5
  cross_validation_PRF(num_folds, featuresets, labels)
  cross_validation_PRF(num_folds, bigram_featuresets, labels)
  cross_validation_PRF(num_folds, POS_featuresets, labels)


  # train classifier and show performance in cross-validation
  train_set, test_set = featuresets[1000:], featuresets[:1000]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print(nltk.classify.accuracy(classifier, test_set))


  #creating a short cut variable name for the bigram association measures
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(all_words)
  bigram_features = finder.nbest(bigram_measures.chi_sq, 500)
  print(bigram_features[:50])




  # bigram_featuresets
  print("bigram_featuresets*********************")
  bigram_featuresets = [(bigram_document_features(d), c) for (d,c) in documents]
  train_set, test_set = bigram_featuresets[1000:], bigram_featuresets[:1000]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print(nltk.classify.accuracy(classifier, test_set))

  # POS tag features
  print("POS tag features*********************")
  POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in documents]
  # number of features for document 0
  len(POS_featuresets[0][0].keys())
  # split into training and test and rerun the classifier
  train_set, test_set = POS_featuresets[1000:], POS_featuresets[:1000]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  nltk.classify.accuracy(classifier, test_set)

  #Subjectivity Lexicon
  # create your own path to the subjclues file
  print("Subjectivity Lexicon*********************")
  SLpath = "SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
  SL = Subjectivity.readSubjectivity(SLpath)
  SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in documents]
  # retrain the classifier using these features
  train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  nltk.classify.accuracy(classifier, test_set)

  # Negation words
  print("NOT_featuresets*********************")
  negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
  # define the feature sets
  NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
  # show the values of a couple of example features
  print(NOT_featuresets[0][0]['V_NOTcare'])
  print(NOT_featuresets[0][0]['V_always'])
  train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  nltk.classify.accuracy(classifier, test_set)
  classifier.show_most_informative_features(30)

  # eval_measures(goldlist, predictedlist)


"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
processkaggle('corpus', 100)
