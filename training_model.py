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
    
    return (g)


# lmao try to follow an algo but clearly it doesn't work
def compute_betweennes(g, nodes):
	N = len(nodes) # number of vertices

	C_B = dict(zip(nodes, [0]*N)) # betweenness centrality

	for s in nodes: # for s in V
		start = time.time()
		print('Computing betweenness of node %s' % s)

		S = [] # init an empty stack
		P = dict(zip(nodes, [[]]*N)) # empty list

		sigma = dict(zip(nodes, [0]*N)) # sigma i.e. number of shortest path
		sigma[s] = 1

		d = dict(zip(nodes, [-1]*N)) # distance of path
		d[s] = 0

		Q = [] # empty queue
		Q.append(s)

		while len(Q) > 0: # while Q is not empty
			v = Q.pop(0)
			S.append(s)
			for _w in g.neighbors(v):
				w = nodes[_w]
				if d[w] < 0:
					Q.append(w)
					d[w] = d[v] + 1
				if d[w] == d[v] + 1:
					sigma[w] = sigma[w] + sigma[v]
					P[w].append(v)
		
		delta = dict(zip(nodes, [0]*N))
		while len(S) > 0:
			w = S.pop()
			for v in P[w]:
				delta[v] = delta[v] + (sigma[v]/sigma[w]) * (1 + delta[w])
				if w != v:
					C_B[w] = C_B[w] + delta[w]

		end = time.time()
		print('--> %.fs' % (end-start))

	return C_B


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

g = build_graph(nodes, edges) # build the graph

print('Number of vertices: %d' % len(g.vs))
print('Number of edges: %d' % len(g.es))

end = time.time()
print('Building the graph takes %.4f s' % ((end-start)%60))


#################################################
# check to see whether graph has multiple edges #
#################################################
start = time.time()

multiple_edges = [e for e in edges if g.is_multiple(e)]
print('Graph has %d multiple edges' % len(multiple_edges))

end = time.time()
print('Checking multiple edges take %.4fs' % (end-start))

# goal: make graph clean of multiple edges (only keep one original)

g.delete_edges([e for e in multiple_edges])
_multiple_edges = [e for e in edges if g.is_multiple(e)]
print('After removal, graph has %d multiple edges' % len(_multiple_edges))


#################
# find pagerank #
#################

start = time.time()

pg = g.pagerank()

end = time.time()
print('Finding page rank of all graph takes %.4fs' % (end-start))

v_pg = zip(nodes, pg)

with open(path_data + 'pagerank_feature.csv', 'wb') as f:
    csv_out = csv.writer(f)
    csv_out.writerow(['name', 'pagerank'])
    for row in v_pg:
        csv_out.writerow(row)

# count = 1
# result = ''
# for v in g.vs:
# 	start = time.time()

# 	pg = g.pagerank(v)
# 	result += v['name'] + ' : ' + str(pg) + '\n'
	
# 	end = time.time()
	
# 	print('Finding page rank of %d-th node takes %.4fs, pg = %.4f' % (count, end-start, pg))

# 	count += 1

# with open(path_data + 'pagerank_info.txt', 'wb') as f:
# 	f.write(result)

#################
# find clusters #
#################
# start = time.time()

# dendogram = g.community_fastgreedy()

# end = time.time()
# print('Finding commnunity by fast greedy takes %.4fs' % (end-start))

# clusters_g = dendogram.as_clustering()
# subg = clusters_g.subgraphs()

# count = 1
# result = ''
# for sg in subg:
# 	print('Subgraph %d has %d nodes' % (count, len(sg.vs)))
# 	result += 'Subgraph ' + str(count) + ' has ' + str(len(sg.vs)) + ' nodes\n'
# 	count += 1

# with open(path_data + 'subgraphs_info.txt', 'wb') as f:
#     f.write(result)

####################################
# compute betweennees by community #
####################################

# sg = subg[3]

# print('Starting to compute betweenness in subgraph of %d nodes' % len(sg.vs))
# count = 1

# for v in sg.vs:
# 	start = time.time()

# 	if sg.degree(v) >= 2:
# 		btw = sg.betweenness(v, cutoff=5)

# 	end = time.time()
# 	print('Computing betweenness of %d-th node takes %.4fs' % (count, end-start))
# 	count += 1

#####################
# compute closeness #
#####################
# start = time.time()

# closeness = [(v, g.closeness(v)) for v in nodes] # compute tuple of (v_id, closeness of v)

# end = time.time()

# print('Computing closeness takes %.4fs' % (end-start))


#################################
# trying to compute betweenness #
#################################
# small_set = [n for n in nodes if g.degree(n) > 1]

# print(len(small_set))

# start = time.time()

# #betweenness = [(v, g.betweenness(v, cutoff=3)) for v in small_set]

# end = time.time()

# print('Compute betweenness for small set of nodes with %d nodes takes %.4fs' % (len(small_set), (end-start)))

####################################
# saving closeness measure in file #
####################################
# with open(path_data + 'closeness_feature.csv', 'wb') as f:
#     csv_out = csv.writer(f)
#     csv_out.writerow(['name', 'closeness'])
#     for row in closeness:
#         csv_out.writerow(row)