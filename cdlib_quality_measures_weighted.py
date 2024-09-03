import networkx as nx
from cdlib.utils import convert_graph_formats
from collections import namedtuple
import numpy as np
import scipy
from cdlib.evaluation.internal.link_modularity import cal_modularity
import Eva
from typing import Callable
from collections import defaultdict
import cdlib.evaluation.fitness


"""
Implement weighted version for some CDLib quality measures

Expansion
Internal density
Cut ratio
Normalized Cut
Avg ODF
Flake ODF

"""

FitnessResult = namedtuple("FitnessResult", "min max score std")
FitnessResult.__new__.__defaults__ = (None,) * len(FitnessResult._fields)


def expansion(graph: nx.Graph, community: object, summary: bool = True, weight: str = None) -> object:
    if weight is None:
        return cdlib.evaluation.fitness.expansion(graph, community, summary)

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ns = len(coms.nodes())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += graph[n][n1][weight]
        try:
            exp = float(edges_outside) / ns
        except:
            exp = 0
        values.append(exp)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def internal_edge_density(
    graph: nx.Graph, community: object, summary: bool = True, weight: str = None
) -> object:
    if weight is None:
        return cdlib.evaluation.fitness.internal_edge_density(graph, community, summary)

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ms = sum([w for u,v,w in coms.edges(data=weight)])
        ns = len(coms.nodes())
        try:
            internal_density = float(ms) / (float(ns * (ns - 1)) / 2)
        except:
            internal_density = 0
        values.append(internal_density)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def cut_ratio(graph: nx.Graph, community: object, summary: bool = True, weight: str = None) -> object:
    if weight is None:
        return cdlib.evaluation.fitness.cut_ratio(graph, community, summary)

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ns = len(coms.nodes())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += graph[n][n1][weight]
        try:
            ratio = float(edges_outside) / (ns * (len(graph.nodes()) - ns))
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values

def normalized_cut(graph: nx.Graph, community: object, summary: bool = True, weight: str = None) -> object:
    if weight is None:
        return cdlib.evaluation.fitness.normalized_cut(graph, community, summary)

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    all_edges_weight = sum([w for u,v,w in graph.edges(data=weight)])

    for com in community.communities:
        coms = nx.subgraph(graph, com)

        ms = sum([w for u,v,w in coms.edges(data=weight)])
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += graph[n][n1][weight]
        try:
            ratio = (float(edges_outside) / ((2 * ms) + edges_outside)) + float(
                edges_outside
            ) / (2 * (all_edges_weight - ms) + edges_outside)
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values

def __out_degree_fraction(g: nx.Graph, coms: list, weight: str) -> list:
    nds = []
    for n in coms:
        nds.append(g.degree(n, weight=weight) - coms.degree(n, weight=weight))
    return nds

def avg_odf(graph: nx.Graph, community: object, summary: bool = True, weight: str = None) -> object:
    if weight is None:
        return cdlib.evaluation.fitness.avg_odf(graph, community, summary)

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        values.append(float(sum(__out_degree_fraction(graph, coms, weight))) / len(coms))

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values


def flake_odf(graph: nx.Graph, community: object, summary: bool = True, weight: str = None) -> object:    
    if weight is None:
        return cdlib.evaluation.fitness.flake_odf(graph, community, summary)

    graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community.communities:
        coms = nx.subgraph(graph, com)
        df = 0
        for n in coms:
            fr = coms.degree(n, weight=weight) - (graph.degree(n, weight=weight) - coms.degree(n, weight=weight))
            if fr < 0:
                df += 1 # TODO: validate if weighted flake odf is based on the count based score or weighted sum score
        score = float(df) / len(coms)
        values.append(score)

    if summary:
        return FitnessResult(
            min=min(values), max=max(values), score=np.mean(values), std=np.std(values)
        )
    return values