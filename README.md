# weighted Graph Hierarchical Agglomerative Clustering (wGHAC)

In this repository the code for wGHAC community detection method is presented.

The main dependency of this method is on Closed Trail (CT) distance. The binary file is available in the closed_trail_distance_binary directory. The binary was build for Unix. For building of binary by yourself I am actually preparing the public repository. You can contact me via an email (petr.prokop@vsb.cz) for other informations.

The wGHAC algorithm uses maximal cliques in a graph as base elements and uses proposed dissimilarities for agglomeration. Dissimilarity depends on the size of the overlap and on the CT distance between vertices.

Short description of included source files:
- **functions.py** contains functions and utilies primarily used for community quality evaluation
- **graph_hierarchical_agglomerative_clustering.py** holds object with algorithm for GHAC calculation
- **run_ghac_community_detection.py** includes example for use of wGHAC on Zachary's karate club network