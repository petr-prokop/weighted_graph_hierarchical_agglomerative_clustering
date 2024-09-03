import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
import pandas as pd
import numpy as np
import itertools
import pickle
import os
import math
import scipy.cluster.hierarchy
import scipy.spatial.distance
import cdlib
import cdlib.algorithms

from graph_hierarchical_agglomerative_clustering import GHACLinkageMethod, GraphAgglomerativeClusteringClosedTrail
import functions

def evaluate_hierarchy(graph:nx.Graph, linkage_matrix:np.ndarray, bases:list, weight_param:str=None, min_distance_in_modularity_calculation:float=0.000, xlim:tuple[int, int]=None, figsize:tuple[int, int]=(36,12), ground_truth_communities:list|None=None, plot_dendrograms:bool=True, ct_distance_matrix:np.ndarray=None):
    dendrogram_modularity_info = dict()
    levels_for_calculation = list()
    distance_vector = linkage_matrix[:, 2].copy()
    linkage_matrix[:, 2] = range(1, linkage_matrix.shape[0]+1)
    
    for level in np.unique(linkage_matrix[:, 2]):
        if distance_vector[int(level-1)] < min_distance_in_modularity_calculation or distance_vector[int(level-1)] < 0:
            continue
        levels_for_calculation.append(level)
    previous_max_distance = distance_vector[0]
    previous_level = 0
    if ground_truth_communities is not None:
        communities_count_hint_min = len(ground_truth_communities) // 2
        communities_count_hint_max = len(ground_truth_communities) * 2
    else:
        communities_count_hint_min = 2
        communities_count_hint_max = 100
    for level in levels_for_calculation:
        distance = distance_vector[int(level-1)]
        comm_list = functions.get_clustering_comm_list(scipy.cluster.hierarchy.fcluster(linkage_matrix, t=level, criterion='distance'), bases)        
        overlapping_communities = functions.merge_bases_into_nodes(comm_list)
        overlapping_communities = functions.drop_small_communities(overlapping_communities, min_size=5)
        if len(overlapping_communities) > communities_count_hint_max or len(overlapping_communities) < communities_count_hint_min:
            continue
        overlapping_communities = functions.postprocess_for_full_cover(overlapping_communities, graph.nodes(), graph)

        cd_evaluation = functions.get_overlapping_evaluation_dict(graph, overlapping_communities, weight_param, ground_truth_communities)
        cd_evaluation['distance'] = distance
        cd_evaluation['level'] = level
        cd_evaluation['communities'] = overlapping_communities

        max_ct_diameter = max(functions.get_communities_ct_diameter(overlapping_communities, ct_distance_matrix))
        cd_evaluation['max_ct_diameter'] = max_ct_diameter
        next_linkage_distance = 0
        for i in range(int(level), len(distance_vector)):
            if distance_vector[i] > 0:
                next_linkage_distance = distance_vector[i]
                break
        cd_evaluation['next_linkage_distance'] = next_linkage_distance
        cd_evaluation['dunn_index'] = next_linkage_distance / max_ct_diameter

        mean_silhouette_scores, max_silhouette_scores = functions.silhouette_score_for_overlapping_communities(overlapping_communities, ct_distance_matrix)
        cd_evaluation['silhouette'] = mean_silhouette_scores
        cd_evaluation['silhouette_maxsi'] = max_silhouette_scores

        dendrogram_modularity_info[level] = cd_evaluation
    
    if len(dendrogram_modularity_info) == 0:
        return None

    if plot_dendrograms:
        best_modularity_distance = max(dendrogram_modularity_info, key=lambda k: dendrogram_modularity_info[k]['modularity_eq'])
        metrics = ['silhouette', 'modularity_overlap', 'conductance_weighted']
        x_ticks = levels_for_calculation
        x_ticks = list(range(53))
        gridspec_kw={'height_ratios': [1]*len(metrics)+[6]}
        figsize = (figsize[0], len(metrics)*1 + 6)
        fig, axs = plt.subplots(len(metrics)+1, sharex=True, figsize=figsize, gridspec_kw=gridspec_kw)
        for e, metric_name in enumerate(metrics):
            ax = fig.get_axes()[e]
            ax.plot(dendrogram_modularity_info.keys(), [dendrogram_modularity_info[k][metric_name] for k in dendrogram_modularity_info.keys()], 'x-')
            ax.set_title(f'{metric_name}')
            ax.xaxis.set_ticks(x_ticks)
            ax.grid(True, axis='both')

            if xlim is not None:
                ax.set_xlim(xlim)
        ax2 = axs[-1]
        scipy.cluster.hierarchy.dendrogram(linkage_matrix, orientation='right', ax=ax2, labels=bases, color_threshold=best_modularity_distance + 0.001)
        plt.xlabel('step')
        ax2.set_title('Dendrogram')
        ax2.xaxis.set_ticks(levels_for_calculation)
        ax2.grid(True, axis='x')
        plt.savefig(f'output_dendrogram.png')
        print('File output_dendrogram.png created.')

    return pd.DataFrame(dendrogram_modularity_info).T.reset_index()

def test_karate():
    graph_filename = 'data/graph_zachary.gml'
    min_base_size = 2
    linkage = GHACLinkageMethod.SINGLE

    # preprocess graph
    graph = nx.read_gml(graph_filename)
    nx.set_edge_attributes(graph, 1, 'weight_uniform')
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_edges_from(list(nx.bridges(graph)))
    graph_gcc = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
    graph_gcc = nx.convert_node_labels_to_integers(graph_gcc, label_attribute='original_label')
    nx.set_edge_attributes(graph_gcc, dict([((u,v),1/w) for u,v,w in graph_gcc.edges(data='weight')]), 'cost')
    
    # compute CT distance matrix
    os.system('mkdir tmp_files')    
    nx.write_gml(graph_gcc, 'tmp_files/graph_gcc.gml')
    nx.write_edgelist(graph_gcc, 'tmp_files/graph_gcc.csv', delimiter=' ', data=['cost'])
    os.system('closed_trail_distance_binary/efficient_suurballe\
                tmp_files/graph_gcc.csv edgelist full\
                tmp_files/graph_gcc_distance_matrix.csv')
    ct_distance_matrix = np.loadtxt('tmp_files/graph_gcc_distance_matrix.csv', dtype=int, delimiter='\t')
    os.system('rm -rf tmp_files')

    # get bases (cliques) for GHAC
    cliques = list()
    for clique in nx.find_cliques(graph_gcc):
        if len(clique) >= min_base_size:
            cliques.append(tuple(sorted([int(node) for node in clique])))
    cliques = list(sorted(cliques, key=len, reverse=False))

    # run GHAC
    print('-'*50)
    print('Running GHAC...')
    ghac = GraphAgglomerativeClusteringClosedTrail(graph_gcc, linkage, ct_distance_matrix, cliques, 'weight_uniform')
    linkage_matrix = ghac.run()

    # evaluate hierarchy, plot dendrogram and show best coverage
    print('-'*50)
    print('Evaluation of hierarchical structure...')
    df_results = evaluate_hierarchy(graph_gcc, linkage_matrix, cliques, weight_param='weight_uniform', ct_distance_matrix=ct_distance_matrix)
    print('-'*50)
    print('The best network covers identified by different modularities:')
    df_best = functions.get_best_split_results(df_results)
    with pd.option_context('display.width', None, 'display.max_colwidth', None):
        print(df_best[['modularity', 'max_val', 'comms_len', 'communities']])

if __name__ == '__main__':
    test_karate()
