import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import cPickle as pickle


from nltk.stem.porter import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances

#from tsne import bh_sne
from gensim.models import Word2Vec
stops = set(["http","www","img","border","home","body","a","about","above","after","again","against","all","am","an",
"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't",
"cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
"herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought",
"our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
"than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're",
"they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
"weren't","what","what's","when","when's""where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
"wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves" ])

train_df = pd.read_csv("train_porter.csv", encoding='utf-8').fillna('')
test_df  = pd.read_csv("test_porter.csv", encoding='utf-8').fillna('')

import re
def calc_cosine_dist(text_a ,text_b, vect):
    text_a=re.sub(r'[^\x00-\x7f]',r' ',text_a)
    text_b=re.sub(r'[^\x00-\x7f]',r' ',text_b)
    return pairwise_distances(vect.transform([text_a]), vect.transform([text_b]), metric='cosine')[0][0]

tfv_orig = TfidfVectorizer(ngram_range=(1,2), min_df=2,stop_words=stops,max_features=2000)
tfv_stem = TfidfVectorizer(ngram_range=(1,2), min_df=2,stop_words=stops,max_features=2000)

tfv_orig.fit(
    list(train_df['question1'].values) + 
    list(test_df['question1'].values)+
    list(train_df['question2'].values) + 
    list(test_df['question2'].values)
) 
tfv_stem.fit(
    list(train_df['question1_porter'].values) + 
    list(test_df['question1_porter'].values)+
    list(train_df['question2_porter'].values) + 
    list(test_df['question2_porter'].values)
) 


path="/home/udit/ipython/notebook/quora/input/input/"
train_cosine = train_df.apply(lambda x:calc_cosine_dist(x['question1'],x['question2'],tfv_orig),axis=1)
pd.to_pickle(train_cosine,path+"train_cosine_dist.pkl")

test_cosine = test_df.apply(lambda x:calc_cosine_dist(x['question1'],x['question2'],tfv_orig),axis=1)
pd.to_pickle(test_cosine,path+"test_cosine_dist.pkl")

print('Generate porter levenshtein_2')
train_porter_cosine = train_df.apply(lambda x:calc_cosine_dist(x['question1_porter'],x['question2_porter'],tfv_stem),axis=1)
test_porter_cosine= test_df.apply(lambda x:calc_cosine_dist(x['question1_porter'],x['question2_porter'],tfv_stem),axis=1)

pd.to_pickle(train_porter_cosine,path+"train_porter_cosine_dist.pkl")
pd.to_pickle(test_porter_cosine,path+"test_porter_cosine_dist.pkl")