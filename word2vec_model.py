import pandas as pd
import word2vec_function as wf
import time
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

model = Word2Vec.load("300features_40minwords_10context")
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
num_features = 300


clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(wf.review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = wf.get_avg_feature_vecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(wf.review_to_wordlist(review, remove_stopwords=True))

testDataVecs = wf.get_avg_feature_vecs(clean_test_reviews, model, num_features)

# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"])

# Test & extract results
result = forest.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

start = time.time()  # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0] / 5)

# Initialize a k-means object and use it to extract centroids
k_means_clustering = KMeans(n_clusters=num_clusters)
idx = k_means_clustering.fit_predict(word_vectors)
# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(model.wv.index2word, idx))
# For the first 10 clusters
for cluster in range(10):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in range(len(word_centroid_map.values())):
        if list(word_centroid_map.values())[i] == cluster:
            words.append(list(word_centroid_map.keys())[i])
    print(words)

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = wf.create_bag_of_centroids(review, word_centroid_map)
    counter += 1
# Repeat for test reviews
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = wf.create_bag_of_centroids(review, word_centroid_map)
    counter += 1
# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators=100)
# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
