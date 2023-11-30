import math
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import cdlib

def merge_bases_into_nodes(communities_list):
    communities_dict = dict()
    for i, community_containing_base in enumerate(communities_list):
        communities_dict[i] = set()
        for base in community_containing_base:
            communities_dict[i] = communities_dict[i] | set(base)

    new_array = [set(sorted([node for node in comp])) for comp in communities_dict.values() if len(comp) > 1]
    return np.unique(new_array) if len(new_array) > 1 else new_array


def get_clustering_comm_list(clustering_result, nodelist_labels):
    comm_list = list()
    for i in range(max(clustering_result)+1):
        comm_list.append(list())
    for node_id, cluster_id in zip(nodelist_labels, clustering_result):
        comm_list[cluster_id].append(node_id)
    return [com for com in comm_list if len(com) > 0]

def modularity_eq(graph, communities):
    q = 0.0
    degrees = dict(graph.degree())
    O = np.zeros((graph.number_of_nodes(), len(communities)))
    
    for k, nds in enumerate(communities):
        O[[int(n) for n in nds], k] = 1
    O = O.sum(1, keepdims=True)
    m = np.sum([v for k, v in degrees.items()])/2
    for community_nodes in communities:
        for nd1, nd2 in itertools.combinations(community_nodes, 2):
            if nd1 == nd2:
                continue
            if graph.has_edge(nd1, nd2):
                e = graph[nd1][nd2]
                wt = 1
            else:
                wt = 0            
            q += ((wt - degrees[nd1]*degrees[nd2]/(2*m))*(1/(O[int(nd1), 0]*O[int(nd2), 0])))
    return q/(2*m)

def modularity_eq_Cao(graph, communities, weight=None):
    q = 0.0
    degrees = dict(graph.degree(weight=weight))    
    U = np.zeros((graph.number_of_nodes(), len(communities)))
    
    for k, nds in enumerate(communities):
        subgraph = nx.subgraph(graph, nds)
        subgraph_degrees = dict(subgraph.degree(weight=weight))
        for nd1 in nds:
            U[int(nd1),k] = subgraph_degrees[int(nd1)]
    U = U/U.sum(1, keepdims=True)
    U = np.nan_to_num(U, 1)

    m = np.sum([v for k, v in degrees.items()])/2
    for k, community_nodes in enumerate(communities):
        for nd1, nd2 in itertools.combinations(community_nodes, 2):
            if graph.has_edge(nd1, nd2):
                e = graph[nd1][nd2]
                wt = e.get(weight, 1)
            else:
                wt = 0            
            q += ((wt - degrees[nd1]*degrees[nd2]/(2*m))*U[nd1,k]*U[nd2,k])
    return q/(2*m)

def ratio_of_unassigned_nodes(graph_relabeled, comm_list):
    O = np.zeros((graph_relabeled.number_of_nodes()+1, len(comm_list)))
    for k, nds in enumerate(comm_list):
        O[[int(n) for n in nds], k] = 1
    O[:,O.sum(axis=0)==1] = 0
    O = O.sum(1, keepdims=True)
    return (graph_relabeled.number_of_nodes() - np.count_nonzero(O)) / graph_relabeled.number_of_nodes()

def get_overlapping_ratio_coef(graph_relabeled, comm_list):
    return sum([len(x) for x in comm_list])/nx.number_of_nodes(graph_relabeled)

def cdlib_communities_quality_check(graph, communities):
    if type(communities) is object:
        communities_cdlib_object = communities
    else:
        communities_cdlib_object = type('obj', (object,), {'communities':communities})
    
    evaluation_dict = dict({
        'conductance': cdlib.evaluation.conductance(graph,communities_cdlib_object),
        'normalized_cut': cdlib.evaluation.normalized_cut(graph,communities_cdlib_object),
    })
    return evaluation_dict

def get_overlapping_evaluation_dict(graph, communities, weight_param, gt_communities=None):
    results = dict()
    cdlib_communitites_obj = type('obj', (object,), {'communities':communities})
    results['comms_len'] = len(communities)
    results['ratio_unassigned'] = ratio_of_unassigned_nodes(graph, communities)
    results['ratio_overlapping'] = get_overlapping_ratio_coef(graph, communities)
    results['modularity_cdlib_modularity_overlap'] = cdlib.evaluation.modularity_overlap(graph, cdlib_communitites_obj, weight_param).score
    return results

def drop_small_communities(communities:list, min_size=3):
    filtered_communities = list()
    for community in communities:
        if len(community) >= min_size:
            filtered_communities.append(community)
    return filtered_communities

def drop_not_merged_bases_from_communities(communities:list, bases:list):
    communities_set = set()
    bases_set = set()
    for community in communities:
        communities_set.add(tuple(sorted(community)))
    for base in bases:
        bases_set.add(tuple(sorted(base)))
    return list(communities_set - bases_set)

def get_full_cover_for_communities(communities:list, nodes:list):
    nodes_set = set(nodes)
    for community in communities:
        for node in community:
            if node in nodes_set:
                nodes_set.remove(node)
    for node in nodes_set:
        communities.append(set([node]))
    return communities


def postprocess_for_full_cover(communities:list, nodes:list, graph: nx.Graph):
    nodes_set = set(nodes)
    for community in communities:
        for node in community:
            if node in nodes_set:
                nodes_set.remove(node)
    # for each missing node vote for community by neighbours
    for node in nodes_set:
        neighbours = set(graph.neighbors(node))
        neighbours_communities = [len(comm & neighbours) for e, comm in enumerate(communities)]
        comm_id_max = np.argmax(neighbours_communities)
        communities[comm_id_max].add(node)
    return communities

def get_best_split_results(df_results):
    best_results_list = list()
    for modularity_measure in ['modularity_cdlib_modularity_overlap']:        
        df_tmp = df_results.sort_values(modularity_measure, ascending=False).reset_index()
        best = df_tmp.loc[0, :].to_dict()
        best['modularity'] = modularity_measure
        best['max_val'] = best[modularity_measure]
        best_results_list.append(best)
    return pd.DataFrame.from_dict(best_results_list)
