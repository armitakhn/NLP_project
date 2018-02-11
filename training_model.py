import random
import csv
import numpy as np
#import igraph
#from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk

# uncomment if haven't downloaded these
#nltk.download('punkt') # for tokenization
#nltk.download('stopwords') # for stopwords

stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
path_data = 'data/'

# read in the testing set
# with open(path_data + "testing_set.txt", "r") as f:
#     reader = csv.reader(f)
#     testing_set  = list(reader)

# testing_set = [element[0].split(" ") for element in testing_set] # separate source and target ID

##################################
# data loading and preprocessing #
##################################

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

# read training set
# with open("training_set.txt", "r") as f:
#     reader = csv.reader(f)
#     training_set  = list(reader)

# training_set = [element[0].split(" ") for element in training_set]

# read the information of each node
with open(path_data + "node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info] # an array of node ID

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info] # get the abstract only
vectorizer = TfidfVectorizer(stop_words="english") # create a new TF-IDF vectorizer
features_TFIDF = vectorizer.fit_transform(corpus) # each row is a node in the order of node_info

## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

#edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

#nodes = IDs

## create empty directed graph
#g = igraph.Graph(directed=True)
 
## add vertices
#g.add_vertices(nodes)
 
## add edges
#g.add_edges(edges)

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select a part (5%) of training set
#to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
#to_keep = random.sample(range(len(training_set)), k=10) # pick up 10 rows
#training_set_reduced = [training_set[i] for i in to_keep]

# # we will use three basic features:

# # number of overlapping words in title
# overlap_title = []

# # temporal distance between the papers
# temp_diff = []

# # number of common authors
# comm_auth = []

# counter = 0
# for i in xrange(len(training_set_reduced)):
#     source = training_set_reduced[i][0]
#     target = training_set_reduced[i][1]
    
#     index_source = IDs.index(source)
#     index_target = IDs.index(target)
    
#     source_info = [element for element in node_info if element[0]==source][0]
#     target_info = [element for element in node_info if element[0]==target][0]
    
# 	# convert to lowercase and tokenize
#     source_title = source_info[2].lower().split(" ")
# 	# remove stopwords
#     source_title = [token for token in source_title if token not in stpwds]
#     source_title = [stemmer.stem(token) for token in source_title]
    
#     target_title = target_info[2].lower().split(" ")
#     target_title = [token for token in target_title if token not in stpwds]
#     target_title = [stemmer.stem(token) for token in target_title]
    
#     source_auth = source_info[3].split(",")
#     target_auth = target_info[3].split(",")
    
#     overlap_title.append(len(set(source_title).intersection(set(target_title))))
#     temp_diff.append(int(source_info[1]) - int(target_info[1]))
#     comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
   
#     counter += 1
#     if counter % 1000 == True:
#         print counter, "training examples processsed"

# # convert list of lists into array
# # documents as rows, unique words as columns (i.e., example as rows, features as columns)
# training_features = np.array([overlap_title, temp_diff, comm_auth]).T

# # scale
# training_features = preprocessing.scale(training_features)

# # convert labels into integers then into column array
# labels = [int(element[2]) for element in training_set_reduced]
# labels = list(labels)
# labels_array = np.array(labels)

# # initialize basic SVM
# classifier = svm.LinearSVC()

# # train
# classifier.fit(training_features, labels_array)

# # test
# # we need to compute the features for the testing set

# overlap_title_test = []
# temp_diff_test = []
# comm_auth_test = []
   
# counter = 0
# for i in xrange(len(testing_set)):
#     source = testing_set[i][0]
#     target = testing_set[i][1]
    
#     source_info = [element for element in node_info if element[0]==source][0]
#     target_info = [element for element in node_info if element[0]==target][0]
    
#     source_title = source_info[2].lower().split(" ")
#     source_title = [token for token in source_title if token not in stpwds]
#     source_title = [stemmer.stem(token) for token in source_title]
    
#     target_title = target_info[2].lower().split(" ")
#     target_title = [token for token in target_title if token not in stpwds]
#     target_title = [stemmer.stem(token) for token in target_title]
    
#     source_auth = source_info[3].split(",")
#     target_auth = target_info[3].split(",")
    
#     overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
#     temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
#     comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
   
#     counter += 1
#     if counter % 1000 == True:
#         print counter, "testing examples processsed"
        
# # convert list of lists into array
# # documents as rows, unique words as columns (i.e., example as rows, features as columns)
# testing_features = np.array([overlap_title_test,temp_diff_test,comm_auth_test]).T

# # scale
# testing_features = preprocessing.scale(testing_features)

# # issue predictions
# predictions_SVM = list(classifier.predict(testing_features))

# # write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
# predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

# with open("improved_predictions.csv","wb") as pred1:
#     csv_out = csv.writer(pred1)
#     for row in predictions_SVM:
#         csv_out.writerow(row)