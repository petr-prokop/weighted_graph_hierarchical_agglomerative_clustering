import networkx as nx
import numpy as np
import enum

class GHACLinkageMethod(enum.Enum):
    SINGLE = 1
    COMPLETE = 2
    AVERAGE = 3

class GraphAgglomerativeClusteringClosedTrail():
    def __init__(self, graph: nx.Graph, ct_linkage_method: GHACLinkageMethod, ct_distance_matrix: np.ndarray, bases: list, weight_attribute=None):
        self.graph = graph
        self.m = nx.number_of_edges(self.graph)
        self.degrees = dict(nx.degree(self.graph))
        self.ct_linkage_method = ct_linkage_method
        self.ct_distance_matrix = ct_distance_matrix
        self.bases = bases
        self.weight_attribute = weight_attribute
        self.wt = None
        if weight_attribute is not None:
            self.wt = sum([w for u,v,w in self.graph.edges(data=weight_attribute)])
        self.reset()
        self.linkage_matrix = None
        
    def reset(self):
        self.clusters_map_of_sets = dict()
        self.clusters_map_of_edges_sets = dict() # this dictionary contains list of edges for conducting subgraph for clusters
        for i, base in enumerate(self.bases):
            self.clusters_map_of_sets[i] = set(base)            
            graph_overlap = nx.subgraph(self.graph, base)
            self.clusters_map_of_edges_sets[i] = set()
            for u,v in nx.edges(graph_overlap):
                self.clusters_map_of_edges_sets[i].add((min(u,v), max(u,v)))
        
    def run(self):
        bases_count = len(self.bases)
        print('Start pairwise distance matrix calculation.')
        distance_matrix = self.calculate_pairwise_distance_matrix()
        print('Calculation finished.')
        linkage_matrix = np.empty((bases_count - 1, 4))
        linkage_clusters_reuse_translation = list(range(bases_count))
        np.fill_diagonal(distance_matrix, 999)
        i = 0
        while i < bases_count - 1:
            if i % 100 == 0:
                print('Agglomeration', i, bases_count - 1)
            tmp = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)            
            m1, m2 = tmp
            
            linkage_matrix[i, 0] = linkage_clusters_reuse_translation[m1]
            linkage_matrix[i, 1] = linkage_clusters_reuse_translation[m2]
            linkage_matrix[i, 2] = distance_matrix[m1, m2]
            linkage_clusters_reuse_translation[m1] = bases_count + i
            
            self.clusters_map_of_sets[m1] = self.clusters_map_of_sets[m1] | self.clusters_map_of_sets[m2]
            self.clusters_map_of_sets[m2] = None
            self.clusters_map_of_edges_sets[m1] = self.clusters_map_of_edges_sets[m1] | self.clusters_map_of_edges_sets[m2]
            self.clusters_map_of_edges_sets[m2] = None
            linkage_matrix[i, 3] = len(self.clusters_map_of_sets[m1])

            for idx in range(bases_count):
                if self.clusters_map_of_sets[idx] is not None and idx != m1:
                    d = self.calculate_ct_method_between_clusters(self.clusters_map_of_sets[m1], self.clusters_map_of_sets[idx], self.clusters_map_of_edges_sets[m1], self.clusters_map_of_edges_sets[idx])
                    if d>0 and distance_matrix[m1, idx] == 997:
                        continue
                    distance_matrix[m1, idx] = d
                    distance_matrix[idx, m1] = d

            distance_matrix[m2, :] = 999
            distance_matrix[:, m2] = 999
            i += 1
        print('Agglomeration finished.')
        self.linkage_matrix = linkage_matrix
        return self.linkage_matrix
            
    def calculate_pairwise_distance_matrix(self):
        bases_count = len(self.bases)
        clusters_distance_matrix = np.zeros((bases_count, bases_count))
        for i in range(bases_count):
            for j in range(i+1, bases_count):
                d = self.calculate_ct_method_between_clusters(self.clusters_map_of_sets[i], self.clusters_map_of_sets[j], self.clusters_map_of_edges_sets[i], self.clusters_map_of_edges_sets[j])
                clusters_distance_matrix[i, j] = d
                clusters_distance_matrix[j, i] = d
        return clusters_distance_matrix

    def calculate_ct_method_between_clusters(self, cluster1, cluster2, edges_list1, edges_list2):     
        intersect = cluster1 & cluster2
        submatrix_indices = np.ix_(list(cluster1 - intersect), list(cluster2 - intersect))
        submatrix = self.ct_distance_matrix[submatrix_indices]        
        if submatrix.size == 0:
            return 0
        d = None
        if self.ct_linkage_method == GHACLinkageMethod.SINGLE:
            d = np.min(submatrix)
        elif self.ct_linkage_method == GHACLinkageMethod.COMPLETE:
            d = np.max(submatrix)
        elif self.ct_linkage_method == GHACLinkageMethod.AVERAGE:
            d = np.average(submatrix)

        if len(intersect) > 0:
            if len(intersect) == 1:
                graph_overlap = nx.subgraph(self.graph, [node for node in intersect])
            else:
                graph_overlap = nx.edge_subgraph(self.graph, edges_list1 & edges_list2) 
            cliques_in_overlap = list(nx.find_cliques(graph_overlap))
            max_clique_size = len(max(cliques_in_overlap, key=len)) if len(cliques_in_overlap) > 0 else 0
            denominator = 1 + max_clique_size
            
            if self.weight_attribute is not None:
                cliques_in_overlap = [clique for clique in cliques_in_overlap if len(clique) == max_clique_size]
                weighted_cliques_list = [sum([w/self.wt for w in nx.get_edge_attributes(nx.subgraph(graph_overlap, clique), name=self.weight_attribute).values()]) for clique in cliques_in_overlap]
                max_overlap_weight = max(weighted_cliques_list) if len(weighted_cliques_list) > 0 else 0
                denominator += max_overlap_weight
            
            d /= denominator
        return d

