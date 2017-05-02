import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack
import numpy
seed=1024
np.random.seed(seed)
path = ""
train = pd.read_csv(path+"train_porter.csv")

fedtype={'am_q1': numpy.float64,
 'am_q2': numpy.float64,
 'are_q1': numpy.float64,
 'are_q2': numpy.float64,
 'can_q1': numpy.float64,
 'can_q2': numpy.float64,
 'could_q1': numpy.float64,
 'could_q2': numpy.float64,
 'did_q1': numpy.float64,
 'did_q2': numpy.float64,
 'do_q1': numpy.float64,
 'do_q2': numpy.float64,
 'does_q1': numpy.float64,
 'does_q2': numpy.float64,
 'had_q1': numpy.float64,
 'had_q2': numpy.float64,
 'has_q1': numpy.float64,
 'has_q2': numpy.float64,
 'have_q1': numpy.float64,
 'have_q2': numpy.float64,
 'how_q1': numpy.float64,
 'how_q2': numpy.float64,
 'is_q1': numpy.float64,
 'is_q2': numpy.float64,
 'may_q1': numpy.float64,
 'may_q2': numpy.float64,
 'might_q1': numpy.float64,
 'might_q2': numpy.float64,
 'q1digits': numpy.float64,
 'q1endwith!': numpy.float64,
 'q1endwith.': numpy.float64,
 'q1endwith?': numpy.float64,
 'q1sentiment': numpy.float64,
 'q1subjectivity': numpy.float64,
 'q2digits': numpy.float64,
 'q2endwith!': numpy.float64,
 'q2endwith.': numpy.float64,
 'q2endwith?': numpy.float64,
 'q2sentiment': numpy.float64,
 'q2subjectivity': numpy.float64,
 'shall_q1': numpy.float64,
 'shall_q2': numpy.float64,
 'should_q1': numpy.float64,
 'should_q2': numpy.float64,
 'was_q1': numpy.float64,
 'was_q2': numpy.float64,
 'were_q1': numpy.float64,
 'were_q2': numpy.float64,
 'what_q1': numpy.float64,
 'what_q2': numpy.float64,
 'when_q1': numpy.float64,
 'when_q2': numpy.float64,
 'where_q1': numpy.float64,
 'where_q2': numpy.float64,
 'which_q1': numpy.float64,
 'which_q2': numpy.float64,
 'who_q1': numpy.float64,
 'who_q2': numpy.float64,
 'why_q1': numpy.float64,
 'why_q2': numpy.float64,
 'will_q1': numpy.float64,
 'will_q2': numpy.float64,
 'would_q1': numpy.float64,
 'would_q2': numpy.float64}

# tfidf
train_question1_tfidf = pd.read_pickle(path+'train_question1_tfidf.pkl')[:]
test_question1_tfidf = pd.read_pickle(path+'test_question1_tfidf.pkl')[:]

train_question2_tfidf = pd.read_pickle(path+'train_question2_tfidf.pkl')[:]
test_question2_tfidf = pd.read_pickle(path+'test_question2_tfidf.pkl')[:]


train_question1_porter_tfidf = pd.read_pickle(path+'train_question1_porter_tfidf.pkl')[:]
test_question1_porter_tfidf = pd.read_pickle(path+'test_question1_porter_tfidf.pkl')[:]

train_question2_porter_tfidf = pd.read_pickle(path+'train_question2_porter_tfidf.pkl')[:]
test_question2_porter_tfidf = pd.read_pickle(path+'test_question2_porter_tfidf.pkl')[:]


train_interaction = pd.read_pickle(path+'train_interaction.pkl')[:].reshape(-1,1)
test_interaction = pd.read_pickle(path+'test_interaction.pkl')[:].reshape(-1,1)

train_porter_interaction = pd.read_pickle(path+'train_porter_interaction.pkl')[:].reshape(-1,1)
test_porter_interaction = pd.read_pickle(path+'test_porter_interaction.pkl')[:].reshape(-1,1)


train_jaccard = pd.read_pickle(path+'train_jaccard.pkl')[:].reshape(-1,1)
test_jaccard = pd.read_pickle(path+'test_jaccard.pkl')[:].reshape(-1,1)

train_porter_jaccard = pd.read_pickle(path+'train_porter_jaccard.pkl')[:].reshape(-1,1)
test_porter_jaccard = pd.read_pickle(path+'test_porter_jaccard.pkl')[:].reshape(-1,1)

train_len = pd.read_pickle(path+"train_len.pkl")
test_len = pd.read_pickle(path+"test_len.pkl")
scaler = MinMaxScaler()
scaler.fit(np.vstack([train_len,test_len]))
train_len = scaler.transform(train_len)
test_len =scaler.transform(test_len)

testquestiontypeFeatures = pd.read_csv("testquestiontypeFeatures.csv",dtype=fedtype)
trainquestiontypeFeatures = pd.read_csv("trainquestiontypeFeatures.csv",dtype=fedtype)
#testquestiontypeFeatures = testquestiontypeFeatures.astype(float)
#trainquestiontypeFeatures = trainquestiontypeFeatures.astype(float)



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
    train_len,
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
    test_len,
    ]).tocsr()


print X.shape
print X_t.shape

skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

dump_svmlight_file(X,y,path+"X_tfidf.svm")
del X
dump_svmlight_file(X_t,np.zeros(X_t.shape[0]),path+"X_t_tfidf.svm")
del X_t

def oversample(X_ot,y,p=0.165):
    pos_ot = X_ot[y==1]
    neg_ot = X_ot[y==0]
    #p = 0.165
    scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
        scale -=1
    neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
    y=np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]]=1.0
    print y.mean()
    return ot,y

X_train,y_train = oversample(X_train.tocsr(),y_train,p=0.165)
X_test,y_test = oversample(X_test.tocsr(),y_test,p=0.165)

X_train,y_train = shuffle(X_train,y_train,random_state=seed)

dump_svmlight_file(X_train,y_train,path+"X_train_tfidf.svm")
dump_svmlight_file(X_test,y_test,path+"X_test_tfidf.svm")

