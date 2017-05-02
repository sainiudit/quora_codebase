import gensim
import numpy as np
embedder = gensim.models.word2vec.KeyedVectors.load_word2vec_format('../lstm/GoogleNews-vectors-negative300.bin', binary=True)

import re
def calc_w2v_sim(question1,question2):
    '''
    Calc w2v similarities and diff of centers of query\title
    '''
    question1=re.sub(r'[^\x00-\x7f]',r' ',question1)
    question2=re.sub(r'[^\x00-\x7f]',r' ',question2)
    a2 = [x for x in question1.lower().split() if x in embedder.vocab]
    b2 = [x for x in question2.lower().split() if x in embedder.vocab]
    if len(a2)>0 and len(b2)>0:
        w2v_sim = embedder.n_similarity(a2, b2)
    else:
        return((-1, -1, np.zeros(300)))
    
    vectorA = np.zeros(300)
    for w in a2:
        vectorA += embedder[w]
    vectorA /= len(a2)

    vectorB = np.zeros(300)
    for w in b2:
        vectorB += embedder[w]
    vectorB /= len(b2)

    vector_diff = (vectorA - vectorB)

    w2v_vdiff_dist = np.sqrt(np.sum(vector_diff**2))
    return w2v_sim, w2v_vdiff_dist, vector_diff
	
train_df = pd.read_csv("train_porter.csv", encoding='utf-8').fillna('')
test_df  = pd.read_csv("test_porter.csv", encoding='utf-8').fillna('')

X_w2v = []
sim_list = []
dist_list = []
for i,row in train_df.iterrows():
    sim, dist, vdiff = calc_w2v_sim(row['question1_porter'],row['question2_porter'])
    X_w2v.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_w2v_tr = np.array(X_w2v)
train_df['w2v_sim'] = np.array(sim_list)
train_df['w2v_dist'] = np.array(dist_list)

path="/home/udit/ipython/notebook/quora/input/input/"
pd.to_pickle(sim_list,path+"train_porter_w2vecsim.pkl")
pd.to_pickle(dist_list,path+"train_porter_w2vecdist.pkl")
pd.to_pickle(X_w2v_tr,path+"train_porter_w2vecvector_diff.pkl")

X_w2v = []
sim_list = []
dist_list = []
for i,row in test_df.iterrows():
    sim, dist, vdiff = calc_w2v_sim(row['question1_porter'],row['question2_porter'])
    X_w2v.append(vdiff)
    sim_list.append(sim)
    dist_list.append(dist)
X_w2v_te = np.array(X_w2v)
test_df['w2v_sim'] = np.array(sim_list)
test_df['w2v_dist'] = np.array(dist_list)

#del embedder
pd.to_pickle(sim_list,path+"test_porter_w2vecsim.pkl")
pd.to_pickle(dist_list,path+"test_porter_w2vecdist.pkl")
#pd.to_pickle(X_w2v_te,path+"test_porter_w2vecvector_diff.pkl")

train_df[['w2v_sim','w2v_dist']].to_csv("wors2vecsim_train.csv",index=False)
test_df[['w2v_sim','w2v_dist']].to_csv("wors2vecsim_test.csv",index=False)