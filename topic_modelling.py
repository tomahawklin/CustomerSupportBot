import pickle as pkl
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

num_clusters = 10
stop_words = list(ENGLISH_STOP_WORDS) + ['flight', 'hey', 'delta', 'airline', 'airlines', 'american', 'air', 'americanair', 'british_airways', 'hi', 'plane', 'https']


def get_center_distance(centers):
	num_center, num_feature = centers.shape
	mat = np.zeros((num_center, num_center))
	for i in range(num_center):
		for j in range(num_center):
			mat[i, j] = sum([t ** 2 for t in (centers[i] - centers[j])])
	return np.sum(mat, axis = 0)


def display_topic(clusters, num_words = 10, stop_words = stop_words):
	for i in clusters:
		text = clusters[i]
		tv = TfidfVectorizer(max_features = 50, stop_words = stop_words)
		tv.fit(text)
		indices = np.argsort(tv.idf_)
		features = tv.get_feature_names()
		top_features = [features[j] for j in indices[:num_words]]
		print('Cluster %d' % i)
		print(top_features)

def get_clusters(X, text, num_clusters = num_clusters):
	kmeans = KMeans(n_clusters = num_clusters).fit(X)
	labels = kmeans.labels_
	centers = kmeans.cluster_centers_
	print(get_center_distance(centers))
	clusters = {}
	n = 0
	for item in labels:
		if item in clusters:
			clusters[item].append(text[n])
		else:
			clusters[item] = [text[n]]
		n += 1
	return clusters


train = pkl.load(open('train.pkl','rb'))

# CountVectorizer cluster
cv = CountVectorizer(ngram_range = (1, 2), min_df = 2, max_features = 50000, stop_words = stop_words)
train_cv = cv.fit_transform(train)

clusters = get_clusters(train_cv, train)
display_topic(clusters)
print([len(clusters[k]) for k in clusters])

# TfidfVectorizer cluster
tv = TfidfVectorizer(ngram_range = (1, 2), min_df = 2, max_features = 50000, stop_words = stop_words)
train_tv = tv.fit_transform(train)

clusters = get_clusters(train_tv, train)
display_topic(clusters)
print([len(clusters[k]) for k in clusters])

# Word2Vec feature cluster
train_tmp = [t.split() for t in train]
model = Word2Vec(train_tmp, window = 100, min_count = 2)
vectors = []
for t in train_tmp:
	mat = [model.wv[item] if item in model.wv else np.zeros(model.wv['As'].shape) for item in t]
	mat = np.array(mat)
	vectors.append(np.sum(mat, axis=0))

vectors = np.array(vectors)
clusters = get_clusters(vectors, train)
display_topic(clusters)
print([len(clusters[k]) for k in clusters])

