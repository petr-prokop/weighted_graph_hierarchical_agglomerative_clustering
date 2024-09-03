import math
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import cdlib
import cdlib_quality_measures_weighted

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

def modularity_eq(graph, communities, weight=None):
    q = 0.0
    degrees = dict(graph.degree(weight=weight))
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
                wt = e[weight] if weight is not None else 1
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

def calculate_weighted_conductance(G, communities_list, weight_param):
    """
    Calculate the weighted conductance for a list of communities within a graph.

    Parameters:
    - G: A networkx graph
    - communities_list: A list of sets, each set contains nodes representing a community within G

    Returns:
    - conductance_list: A list of weighted conductance values for each community
    """
    if not isinstance(G, nx.Graph):
        raise ValueError("G must be a networkx Graph.")
    if not isinstance(communities_list, list) or not all(isinstance(c, set) for c in communities_list):
        raise ValueError("communities_list must be a list of sets of nodes.")

    conductance_list = []
    for community in communities_list:
        intra_community_weight = sum(G[u][v][weight_param] for u, v in G.subgraph(community).edges())
        inter_community_weight = sum(G[u][v][weight_param] for u in community for v in G.neighbors(u) if v not in community)
        # community_weighted_degree = sum(G.degree(node, weight=weight_param) for node in community)

        # Avoid division by zero
        try:
            weighted_conductance = inter_community_weight / (2*intra_community_weight + inter_community_weight)
            conductance_list.append(weighted_conductance)
        except:
            conductance_list.append(0)

    return np.mean(conductance_list)

def cdlib_communities_quality_check(graph, communities, weight_param):
    if type(communities) is object:
        communities_cdlib_object = communities
    else:
        communities_cdlib_object = type('obj', (object,), {'communities':communities})
    
    evaluation_dict = dict({
        'conductance': cdlib.evaluation.conductance(graph,communities_cdlib_object).score,
        'expansion': cdlib.evaluation.expansion(graph,communities_cdlib_object).score,    
        'internal_edge_density': cdlib.evaluation.internal_edge_density(graph,communities_cdlib_object).score,
        'cut_ratio': cdlib.evaluation.cut_ratio(graph,communities_cdlib_object).score,
        'normalized_cut': cdlib.evaluation.normalized_cut(graph,communities_cdlib_object).score,
        'flake_odf': cdlib.evaluation.flake_odf(graph,communities_cdlib_object).score,
        'avg_odf': cdlib.evaluation.avg_odf(graph,communities_cdlib_object).score,
        'scaled_density': cdlib.evaluation.scaled_density(graph,communities_cdlib_object).score,
        'avg_transitivity': cdlib.evaluation.avg_transitivity(graph,communities_cdlib_object).score,

        'expansion_uniform': cdlib_quality_measures_weighted.expansion(graph,communities_cdlib_object, weight='weight_uniform').score,
        'internal_edge_density_uniform': cdlib_quality_measures_weighted.internal_edge_density(graph,communities_cdlib_object, weight='weight_uniform').score,
        'cut_ratio_uniform': cdlib_quality_measures_weighted.cut_ratio(graph,communities_cdlib_object, weight='weight_uniform').score,
        'normalized_cut_uniform': cdlib_quality_measures_weighted.normalized_cut(graph,communities_cdlib_object, weight='weight_uniform').score,
        'flake_odf_uniform': cdlib_quality_measures_weighted.flake_odf(graph,communities_cdlib_object, weight='weight_uniform').score,
        'avg_odf_uniform': cdlib_quality_measures_weighted.avg_odf(graph,communities_cdlib_object, weight='weight_uniform').score,
        
        'expansion_weight': cdlib_quality_measures_weighted.expansion(graph,communities_cdlib_object, weight=weight_param).score,
        'internal_edge_density_weight': cdlib_quality_measures_weighted.internal_edge_density(graph,communities_cdlib_object, weight=weight_param).score,
        'cut_ratio_weight': cdlib_quality_measures_weighted.cut_ratio(graph,communities_cdlib_object, weight=weight_param).score,
        'normalized_cut_weight': cdlib_quality_measures_weighted.normalized_cut(graph,communities_cdlib_object, weight=weight_param).score,
        'flake_odf_weight': cdlib_quality_measures_weighted.flake_odf(graph,communities_cdlib_object, weight=weight_param).score,
        'avg_odf_weight': cdlib_quality_measures_weighted.avg_odf(graph,communities_cdlib_object, weight=weight_param).score,
    })
    return evaluation_dict

def get_overlapping_evaluation_dict(graph, communities, weight_param, gt_communities=None):
    results = dict()
    cdlib_communitites_obj = type('obj', (object,), {'communities':communities})
    cdlib_communitites_complete_obj = type('obj', (object,), {'communities':get_full_cover_for_communities(communities, graph.nodes())})
    results['comms_len'] = len(communities)
    results['ratio_overlapping'] = get_overlapping_ratio_coef(graph, communities)
    results['modularity_eq'] = modularity_eq(graph, communities, weight=weight_param)
    results['modularity_eq_uniform'] = modularity_eq(graph, communities, weight='weight_uniform')
    results['conductance_weighted'] = calculate_weighted_conductance(graph, cdlib_communitites_complete_obj.communities, weight_param)
    results['conductance_weighted_uniform'] = calculate_weighted_conductance(graph, cdlib_communitites_complete_obj.communities, 'weight_uniform')
    results['modularity_overlap'] = cdlib.evaluation.modularity_overlap(graph, cdlib_communitites_obj, weight_param).score
    results['modularity_overlap_uniform'] = cdlib.evaluation.modularity_overlap(graph, cdlib_communitites_obj, 'weight_uniform').score

    if gt_communities is not None:
        gt_communities_obj = type('obj', (object,), {'communities':list(gt_communities)})
        results['onmi_1'] = cdlib.evaluation.overlapping_normalized_mutual_information_LFK(cdlib_communitites_obj, gt_communities_obj).score
        results['onmi_2'] = cdlib.evaluation.overlapping_normalized_mutual_information_MGH(cdlib_communitites_obj, gt_communities_obj).score
        results['f1'] = cdlib.evaluation.f1(cdlib_communitites_obj, gt_communities_obj).score
        results['nf1'] = cdlib.evaluation.nf1(cdlib_communitites_obj, gt_communities_obj).score
        results['omega'] = cdlib.evaluation.omega(cdlib_communitites_complete_obj, gt_communities_obj).score

        on_ratio, om_ratio, f1 = calculate_overlapping_quality(graph, communities, gt_communities)
        results['on_ratio'] = on_ratio
        results['om_ratio'] = om_ratio
        results['on_ratio_1_diff'] = np.abs(1-on_ratio)
        results['om_ratio_1_diff'] = np.abs(1-om_ratio)
        results['f1_nodes'] = f1
    cdlib_results_dict = cdlib_communities_quality_check(graph, communities, weight_param)
    results.update(cdlib_results_dict)

    return results


def calculate_overlapping_quality(graph, communities, gt_communities):
    detected_node_assigment_counts = np.zeros(nx.number_of_nodes(graph))
    gt_node_assigment_counts = np.zeros(nx.number_of_nodes(graph))
    for comm in communities:
        for node in comm:
            detected_node_assigment_counts[node] += 1
    for comm in gt_communities:
        for node in comm:
            gt_node_assigment_counts[node] += 1
    
    on_ratio = np.sum(detected_node_assigment_counts > 1) / np.sum(gt_node_assigment_counts > 1)        
    om_ratio = np.mean(detected_node_assigment_counts) / np.mean(gt_node_assigment_counts)
    tp = np.sum((detected_node_assigment_counts > 1) & (gt_node_assigment_counts > 1))
    if tp == 0:
        f1 = 0
    else:
        precision = tp / np.sum(detected_node_assigment_counts > 1)
        recall = tp / np.sum(gt_node_assigment_counts > 1)
        f1 = 2 * (precision * recall) / (precision + recall)
    return on_ratio, om_ratio, f1

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
    missing_node_community_assigment = dict()
    for node in nodes_set:
        neighbours = set(graph.neighbors(node))
        neighbours_communities = [len(comm & neighbours) for e, comm in enumerate(communities)]
        if any([c > 1 for c in neighbours_communities]):
            comm_id_max = np.argmax(neighbours_communities)
            missing_node_community_assigment[node] = comm_id_max
        else:
            missing_node_community_assigment[node] = -1
    for node in nodes_set:
        if missing_node_community_assigment[node] != -1:
            communities[missing_node_community_assigment[node]].add(node)
        else:
            communities.append(set([node]))
    return communities

def get_best_split_results(df_results):
    best_results_list = list()
    for modularity_measure in ['modularity_eq']:        
        df_tmp = df_results.sort_values(modularity_measure, ascending=False).reset_index()
        best = df_tmp.loc[0, :].to_dict()
        best['modularity'] = modularity_measure
        best['max_val'] = best[modularity_measure]
        best_results_list.append(best)
    return pd.DataFrame.from_dict(best_results_list)

def get_communities_ct_diameter(communities, ct_distance_matrix):
    communities_diameter = list()
    for community in communities:
        communities_diameter.append(np.max(ct_distance_matrix[np.ix_(list(community), list(community))]))
    return communities_diameter

def silhouette_score_for_overlapping_communities(communities, ct_distance_matrix):
    node_communities_belonging = dict()
    node_silhouette_scores = dict()

    for i, community in enumerate(communities):
        for node in community:
            if node not in node_communities_belonging:
                node_communities_belonging[node] = list()
            node_communities_belonging[node].append(i)

    for node, communities_belonging in node_communities_belonging.items():
        node_silhouette_scores[node] = list()
        b = None
        for community_beloning in communities_belonging:
            if len(communities[community_beloning]) == 1:
                s = 0.0
                node_silhouette_scores[node].append(s)
                continue

            if b is None:
                all_b = list()
                for e, outer_community in enumerate(communities):
                    if e in communities_belonging:
                        continue
                    b_tmp = np.mean([ct_distance_matrix[node, n] for n in outer_community])
                    all_b.append(b_tmp)
                if len(all_b) == 0:
                    b = 0
                else:
                    b = np.min(all_b)

            a = np.mean([ct_distance_matrix[node, n] for n in communities[community_beloning] if n != node]) if len(communities[community_beloning]) > 1 else 0
            s = (b - a) / max(a, b)
            node_silhouette_scores[node].append(s)
    
    mean_silhouette_scores = np.mean([np.mean(v) for k, v in node_silhouette_scores.items()])
    max_silhouette_scores = np.mean([np.max(v) for k, v in node_silhouette_scores.items()])
    return mean_silhouette_scores, max_silhouette_scores