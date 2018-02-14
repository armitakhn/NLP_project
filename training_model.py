#######################
# importing libraries #
#######################

# import time to find consuming steps
import time
start = time.time()

# utility libraries
import numpy as np
import igraph as ig
import csv
from sklearn import preprocessing as pre
import re

end = time.time()
print('Loading libraries takes %.4f s' % (end-start))


#####################
# utility functions #
#####################

def build_graph(nodes, edges):
    '''
    Build a graph using igraph library
    
    Parameters
    ----------
    nodes: a list of nodes
    edges: a list of tuples (source, target)
    
    Returns
    -------
    a graph g
    '''
    g = ig.Graph(directed=False) # create an undirected graph
    g.add_vertices(nodes) # add nodes
    g.add_edges(edges) # add edges
    
    betweenness = []
    counter = 1
    for v in g.vs:
    	start = time.time()
    	betweenness.append((v['name'], g.betweenness(v, directed=False, cutoff=3)))
    	end = time.time()

    	print('Computed betweenness of %d-th node, ID %s, takes %.4f s' %(counter, v['name'], (end-start)))
    	counter += 1

    # betweenness = [(v['name'], g.betweenness(v, directed=False, cutoff=3)) for v in g.vs]

    #closeness = [(v['name'], g.closeness(v)) for v in g.vs]
    
    return (g, betweenness)


##################################
# data loading and preprocessing #
##################################

path_data = 'data/' # path to the data
path_submission = 'submission/' # path to submission files


# ====== read in node informations ====== #
start = time.time()

with open(path_data + 'node_information.csv', 'r') as f:
    reader = csv.reader(f)
    node_info = list(reader)

end = time.time()
print('Reading node information takes %.4f s' % (end-start))


# ====== read training data as str ====== #
start = time.time()

training = np.genfromtxt(path_data + 'training_set.txt', dtype=str)

end = time.time()
print('Reading training set takes %.4f s' % (end-start))


######################
# building the graph #
######################
start = time.time()

edges = [(element[0], element[1]) for element in training if int(element[2]) == 1] # extract all the edges
nodes = [element[0] for element in node_info] # extract all the vertices

result = build_graph(nodes, edges) # build the graph
g = result[0]
betweenness = result[1]

print('Number of vertices: %d' % len(g.vs))
print('Number of edges: %d' % len(g.es))

end = time.time()
print('Building the graph and computing betweenness take %.4f minutes' % ((end-start)%60))

####################################
# saving closeness measure in file #
####################################
with open(path_data + 'betweenness_feature.csv', 'wb') as f:
    csv_out = csv.writer(f)
    csv_out.writerow(['name', 'betweenness'])
    for row in betweenness:
        csv_out.writerow(row)