# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:23:59 2017

@author: mariosm
"""
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy import sparse as ssp
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import distance

stop_words = stopwords.words('english')

# stops = set(stopwords.words("english"))
stops = set(
    ["http", "www", "img", "border", "home", "body", "a", "about", "above", "after", "again", "against", "all", "am",
     "an",
     "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
     "but", "by", "can't",
     "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
     "each", "few", "for", "from",
     "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
     "here", "here's", "hers",
     "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
     "isn't", "it", "it's", "its",
     "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
     "only", "or", "other", "ought",
     "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
     "shouldn't", "so", "some", "such",
     "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
     "they", "they'd", "they'll", "they're",
     "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
     "we'll", "we're", "we've", "were",
     "weren't", "what", "what's", "when", "when's""where", "where's", "which", "while", "who", "who's", "whom", "why",
     "why's", "with", "won't", "would",
     "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])
porter = PorterStemmer()
snowball = SnowballStemmer('english')

weights = {}


def fromsparsetofile(filename, array, deli1=" ", deli2=":", ytarget=None):
    zsparse = csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))
    print(" indptr lenth %d" % (len(indptr)))

    f = open(filename, "w")
    counter_row = 0
    for b in range(0, len(indptr) - 1):
        # if there is a target, print it else , print nothing
        if ytarget != None:
            f.write(str(ytarget[b]) + deli1)

        for k in range(indptr[b], indptr[b + 1]):
            if (k == indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k], deli2, -1))
                else:
                    f.write("%d%s%f" % (indices[k], deli2, data[k]))
            else:
                if np.isnan(data[k]):
                    f.write("%s%d%s%f" % (deli1, indices[k], deli2, -1))
                else:
                    f.write("%s%d%s%f" % (deli1, indices[k], deli2, data[k]))
        f.write("\n")
        counter_row += 1
        if counter_row % 10000 == 0:
            print(" row : %d " % (counter_row))
    f.close()


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=5000.0, min_count=2.0):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)


def word_shares(row, wei, stop):
    q1 = set(str(row['question1']).lower().split())
    q1words = q1.difference(stop)
    if len(q1words) == 0:
        return '0:0:0:0:0'

    q2 = set(str(row['question2']).lower().split())
    q2words = q2.difference(stop)
    if len(q2words) == 0:
        return '0:0:0:0:0'

    q1stops = q1.intersection(stop)
    q2stops = q2.intersection(stop)

    shared_words = q1words.intersection(q2words)
    # print(len(shared_words))
    shared_weights = [wei.get(w, 0) for w in shared_words]
    total_weights = [wei.get(w, 0) for w in q1words] + [wei.get(w, 0) for w in q2words]

    R1 = np.sum(shared_weights) / np.sum(total_weights)  # tfidf share
    R2 = float(len(shared_words)) / (float(len(q1words)) + float(len(q2words)))  # count share
    R31 = float(len(q1stops)) / float(len(q1words))  # stops in q1
    R32 = float(len(q2stops)) / float(len(q2words))  # stops in q2
    return '{}:{}:{}:{}:{}'.format(R1, R2, float(len(shared_words)), R31, R32)


def stem_str(x, stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]", " ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x


def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) * 1.0 / len(a)


def str_abs_diff_len(str1, str2):
    return abs(len(str1) - len(str2))


def str_len(str1):
    return len(str(str1))


def char_len(str1):
    str1_list = set(str(str1).replace(' ', ''))
    return len(str1_list)


def word_len(str1):
    str1_list = str1.split(' ')
    return len(str1_list)


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) * 1.0 / (len(q1words) + len(q2words))
    return R


def str_jaccard(str1, str2):
    str1_list = str1.split(" ")
    str2_list = str2.split(" ")
    res = distance.jaccard(str1_list, str2_list)
    return res


# shortest alignment
def str_levenshtein_1(str1, str2):
    # str1_list = str1.split(' ')
    # str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2, method=1)
    return res


# longest alignment
def str_levenshtein_2(str1, str2):
    # str1_list = str1.split(' ')
    # str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2, method=2)
    return res


def str_sorensen(str1, str2):
    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.sorensen(str1_list, str2_list)
    return res


def main():
    path = ""
    ###################  generate_svm_format_tfidf.py #################
    train = pd.read_csv(path + "train_porter.csv")

    train_question1_tfidf = pd.read_pickle(path + 'train_question1_tfidf.pkl')[:]
    test_question1_tfidf = pd.read_pickle(path + 'test_question1_tfidf.pkl')[:]

    train_question2_tfidf = pd.read_pickle(path + 'train_question2_tfidf.pkl')[:]
    test_question2_tfidf = pd.read_pickle(path + 'test_question2_tfidf.pkl')[:]

    # train_question1_porter_tfidf = pd.read_pickle(path+'train_question1_porter_tfidf.pkl')[:]
    # test_question1_porter_tfidf = pd.read_pickle(path+'test_question1_porter_tfidf.pkl')[:]

    # train_question2_porter_tfidf = pd.read_pickle(path+'train_question2_porter_tfidf.pkl')[:]
    # test_question2_porter_tfidf = pd.read_pickle(path+'test_question2_porter_tfidf.pkl')[:]


    train_interaction = pd.read_pickle(path + 'train_interaction.pkl')[:].reshape(-1, 1)
    test_interaction = pd.read_pickle(path + 'test_interaction.pkl')[:].reshape(-1, 1)

    train_interaction = np.nan_to_num(train_interaction)
    test_interaction = np.nan_to_num(test_interaction)

    train_porter_interaction = pd.read_pickle(path + 'train_porter_interaction.pkl')[:].reshape(-1, 1)
    test_porter_interaction = pd.read_pickle(path + 'test_porter_interaction.pkl')[:].reshape(-1, 1)

    train_porter_interaction = np.nan_to_num(train_porter_interaction)
    test_porter_interaction = np.nan_to_num(test_porter_interaction)

    train_jaccard = pd.read_pickle(path + 'train_jaccard.pkl')[:].reshape(-1, 1)
    test_jaccard = pd.read_pickle(path + 'test_jaccard.pkl')[:].reshape(-1, 1)

    train_jaccard = np.nan_to_num(train_jaccard)
    test_jaccard = np.nan_to_num(test_jaccard)

    train_porter_jaccard = pd.read_pickle(path + 'train_porter_jaccard.pkl')[:].reshape(-1, 1)
    test_porter_jaccard = pd.read_pickle(path + 'test_porter_jaccard.pkl')[:].reshape(-1, 1)

    train_jaccard = np.nan_to_num(train_jaccard)
    test_porter_jaccard = np.nan_to_num(test_porter_jaccard)

    train_len = pd.read_pickle(path + "train_len.pkl")
    test_len = pd.read_pickle(path + "test_len.pkl")

    train_len = np.nan_to_num(train_len)
    test_len = np.nan_to_num(test_len)

    scaler = MinMaxScaler()
    scaler.fit(np.vstack([train_len, test_len]))
    train_len = scaler.transform(train_len)
    test_len = scaler.transform(test_len)

    testquestiontypeFeatures = pd.read_csv("testquestiontypeFeatures.csv")
    trainquestiontypeFeatures = pd.read_csv("trainquestiontypeFeatures.csv")
    testquestiontypeFeatures = testquestiontypeFeatures.astype(float)
    trainquestiontypeFeatures = trainquestiontypeFeatures.astype(float)

    abhitrain = pd.read_csv("train_features.csv")
    abhitest = pd.read_csv("test_features.csv")
    abhife = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
              'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'wmd', 'norm_wmd', 'cityblock_distance',
              'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec',
              'skew_q2vec', 'kur_q1vec', 'kur_q2vec']


    X = ssp.hstack([
        csr_matrix(abhitrain[abhife]),
        csr_matrix(trainquestiontypeFeatures.values),
        train_question1_tfidf,
        train_question2_tfidf,
        train_interaction,
        train_porter_interaction,
        train_jaccard,
        train_porter_jaccard,
        train_len
    ]).tocsr()

    y = train['is_duplicate'].values[:]

    X_t = ssp.hstack([
        csr_matrix(abhitest[abhife]),
        csr_matrix(testquestiontypeFeatures.values),
        test_question1_tfidf,
        test_question2_tfidf,
        test_interaction,
        test_porter_interaction,
        test_jaccard,
        test_porter_jaccard,
        test_len
    ]).tocsr()

    print X.shape
    print X_t.shape

    fromsparsetofile(path + "x_tfidf.svm", X, deli1=" ", deli2=":", ytarget=y)
    del X
    fromsparsetofile(path + "x_t_tfidf.svm", X_t, deli1=" ", deli2=":", ytarget=None)
    del X_t

    print ("done!")

if __name__ == "__main__":
    main()
