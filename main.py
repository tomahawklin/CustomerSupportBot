import pandas as pd
import re
import numpy as np
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
import random
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


df = pd.read_csv('twcs.csv')
df['created_at'] = df['created_at'].apply(lambda x:dt.datetime.strptime(x,'%a %b %d %H:%M:%S +0000 %Y'))

class tweet(object):
    """
    An object that holds a single tweet
    """
    def __init__(self, df_row):
        self.tweet_id = df_row.tweet_id
        self.author_id = df_row.author_id
        self.created_at = df_row.created_at
        self.text = df_row.text
        self.response_tweet_id = df_row.response_tweet_id
        self.in_response_to_tweet_id = df_row.in_response_to_tweet_id

class dialogue(object):
    """
    A structured object that organizes dialogue between customer and company agent
    """
    def __init__(self, init_tweet):
        self.tweets = [init_tweet]
        self.ids = {init_tweet.tweet_id}
        self.company = None

class vector_model(object):
    """
    Object holding item to index map and item vectors
    """
    def __init__(self, item_map, vectors):
        self.item_map = item_map
        self.vectors = vectors# / np.linalg.norm(vectors, axis = 1, ord = 2)[:,None]
        self.num_items = vectors.shape[0]
        self.dim = vectors.shape[1]
        self.inv_map = {v: k for k, v in self.item_map.items()}
    
    def get_nearest(self, candidate_vector, threshold = 7, topk = 1, ex_self = True):
        if candidate_vector.shape != (1, self.dim):
            candidate_vector = candidate_vector.reshape(1, self.dim)
        score = self.vectors.dot(candidate_vector.T)
        flag = max(score) > threshold
        if score.shape != (self.num_items):
            score = score.reshape(self.num_items)
        top_index = score.argsort()[::-1][:topk].tolist()
        return flag[0], [self.inv_map[i] for i in top_index]

def print_tweets(dialogue, show_id = True):
    """
    Helper function to print tweets in a dialogue
    """
    tweets = sorted(dialogue.tweets, key=lambda k: k.created_at) 
    for t in tweets:
        if show_id:
            print('tweet_id: %s, date: %s, author: %s, text: %s, response_tweet_id: %s, in_response_to_tweet_id: %s' % (t.tweet_id, str(t.created_at), t.author_id, t.text, str(t.response_tweet_id), str(t.in_response_to_tweet_id)))
        else:
            print('date: %s, author: %s, text: %s' % (str(t.created_at), t.author_id, t.text))


candidate = {}
heap = []

for i in df.index.values:
    if i % 100000 == 0:
        print('finished %d rows' % i)
    t = tweet(df.loc[i])
    key = t.in_response_to_tweet_id
    if np.isnan(key):
        heap.append(dialogue(t))
    else:
        if key in candidate:
            candidate[key].append(t)
        else:
            candidate[key] = [t]

# These two numbers should equal
print(len(heap) + sum([len(candidate[k]) for k in candidate]))
print(df.shape[0])

# Assign candidate tweets to dialogues
for i in range(len(heap)):
    h = heap[i]
    waiting = {h.tweets[0].tweet_id}
    while len(waiting) > 0:
        tmp_waiting = []
        for idx in waiting:
            try:
                tmp_list = candidate[float(idx)]
                h.tweets += tmp_list
                h.ids.update([t.tweet_id for t in tmp_list])
                tmp_waiting += [t.tweet_id for t in tmp_list]
                del candidate[float(idx)]
            except:
                pass
        waiting = set(tmp_waiting)

names = [t for t in set(df['author_id'].tolist()) if re.search('[a-zA-Z]', t)]
# Drop these cases
heap = [h for h in heap if len(set([t.author_id for t in h.tweets if t.author_id in names])) == 1]
# Assign company label to each dialogue
for h in heap:
    h.company = [t.author_id for t in h.tweets if t.author_id in names][0]

airlines = [h for h in heap if h.company in ['Delta', 'AmericanAir', 'British_Airways']]
airlines = [d for d in airlines if len(d.tweets) == 2]
data = []
pool = set()
for d in airlines:
    if any(s in d.tweets[0].text.lower() for s in ['how', 'when', 'what', 'where', 'who', 'which', '?']):
        if d.tweets[0].text not in pool:
            data.append(d)
            pool.add(d.tweets[0].text)
        else:
            continue

#del airlines
#del df
#del heap
#del candidate

train = []
test = []
for d in data:
    if random.random() > 0.9:
        test.append(d.tweets[0].text)
    else:
        train.append(d.tweets[0].text)

f = open('data.pkl', 'wb')
pkl.dump(data, f)
f.close()
f = open('train.pkl', 'wb')
pkl.dump(train, f)
f.close()
f = open('test.pkl', 'wb')
pkl.dump(test, f)
f.close()


stop_words = list(ENGLISH_STOP_WORDS) + ['flight', 'hey', 'delta', 'airline', 'airlines', 'american', 'air', 'americanair', 'british_airways', 'hi', 'plane', 'https']
cv = CountVectorizer(ngram_range = (1, 2), min_df = 2, max_features = 50000, stop_words = stop_words)
vectors = cv.fit_transform(train).toarray()
item_map = dict()
for t in train:
    item_map[t] = len(item_map)

cv_model = vector_model(item_map, vectors)
f = open('cv_model_new.pkl', 'wb')
pkl.dump(cv_model, f)
f.close()

# tfidf model is not consistent in threshold experiment
tv = TfidfVectorizer(ngram_range = (1, 2), stop_words = 'english', max_features = 30000)
vectors = tv.fit_transform(train).toarray()
item_map = dict()
for t in train:
    item_map[t] = len(item_map)

tv_model = vector_model(item_map, vectors)
f = open('tv_model.pkl', 'wb')
pkl.dump(tv_model, f)
f.close()

qa_map = dict()
for d in data:
	qa_map[d.tweets[0].text] = d.tweets[1].text

def test_model(model, vectorizer, outfile, test = test, topk = 2, qa_map = qa_map, threshold = 5):
    filename = outfile if '.txt' in outfile else outfile + '.txt'
    f = open(filename, 'w')
    for idx in range(len(test)):
        test_q = test[idx]
        flag, match = model.get_nearest(vectorizer.transform([test_q]).toarray(), topk = topk, threshold = threshold)
        f.write('Test question: %d \n' % idx)
        f.write('\n')
        f.write(test_q + '\n')
        f.write('\n')
        f.write(qa_map[test_q] + '\n')
        f.write('\n')
        if flag:
            f.write('**********SUCCESS**********\n')
        else:
            f.write('**********FAILURE**********\n')
        f.write('\n')
        for i in range(len(match)):
            f.write('Top %d \n' % (i + 1))
            f.write('\n')
            f.write(match[i] + '\n')
            f.write('\n')
            f.write(qa_map[match[i]] + '\n')
            f.write('\n')
    f.close()

def unit_test(model, vectorizer, test = test, topk = 3, qa_map = qa_map, threshold = 5):
    for idx in range(len(test)):
        test_q = test[idx]
        flag, match = model.get_nearest(vectorizer.transform([test_q]).toarray(), threshold = threshold, topk = topk)
        print('Test question %d \n' % idx)
        print(test_q + '\n')
        print(qa_map[test_q] + '\n')
        if flag:
            print('**********SUCCESS**********\n')
        else:
            print('**********FAILURE**********\n')
        for i in range(len(match)):
            print('Top %d \n' % (i + 1))
            print(match[i] + '\n')
            print(qa_map[match[i]] + '\n')

def test_threshold(vectors, v):
    cnt_test = 0
    cnt_train = 0
    for t in train:
        flag, _ = cv_model.get_nearest(cv.transform([t]).toarray(), topk = 2)
        cnt_train = cnt_train + 1 if flag else cnt_train
    for t in test:
        flag, _ = cv_model.get_nearest(cv.transform([t]).toarray(), topk = 2)

    score = vectors.dot(v.T).flatten()
    score.sort()
    return score[-3:]

test_model(cv_model, cv, 'cv_test')
test_model(tv_model, tv, 'tv_test')

train_tmp = [t.split() for t in train]
model = Word2Vec(train_tmp, window = 100, min_count = 2)
vectors = []
for t in train_tmp:
    mat = [model.wv[item] if item in model.wv else np.zeros(model.wv['As'].shape) for item in t]
    mat = np.array(mat)
    vectors.append(np.sum(mat, axis=0))

vectors = np.array(vectors)
w2v_model = vector_model(item_map, vectors)
f = open('w2v_model.pkl', 'wb')
pkl.dump(w2v_model, f)
f.close()

def unit_test_w2v(model, wv, test = test, topk = 1, qa_map = qa_map, threshold = 5):
    for idx in range(len(test)):
        test_q = test[idx]
        vector = [wv[item] if item in wv else np.zeros(wv['As'].shape) for item in test_q.split()]
        vector = np.array(vector)
        vector = np.sum(vector, axis = 0)
        flag, match = model.get_nearest(vector, threshold = threshold, topk = topk)
        print('Test question %d \n' % idx)
        print(test_q + '\n')
        print(qa_map[test_q] + '\n')
        if flag:
            print('**********SUCCESS**********\n')
        else:
            print('**********FAILURE**********\n')
        for i in range(len(match)):
            print('Top %d \n' % (i + 1))
            print(match[i] + '\n')
            print(qa_map[match[i]] + '\n')