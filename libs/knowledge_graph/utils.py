import os
import pandas as pd
import networkx as nx


def load_knowledge_graph(kg_nodes_path, kg_edges_path):

    G = nx.Graph()
    id_to_node = load_nodes(kg_nodes_path)
    edges = load_edges(kg_edges_path)

    G.add_nodes_from(id_to_node.items())

    for id_pair in edges:
        G.add_edge(*id_pair)

    return G


def load_nodes(kg_nodes_path: str):
    """

    """
    # The first two lines are meta information
    df = pd.read_csv(kg_nodes_path, delimiter="\t", skiprows=2)

    id_to_node = {}
    for _, row in df.iterrows():
        tmp = row["ROW"].split("|")[:-1]
        tmp = [x.strip() for x in tmp]
        description = " ".join([x for x in tmp if x != ""])
        uid = row["UID"]
        table = row["TABLE"]
        id_to_node[uid] = {
            "table": table,
            "description": description,
        }
    return id_to_node


def load_edges(kg_edges_path: str):
    """

    """
    df = pd.read_csv(kg_edges_path, delimiter="\t", index_col=0)
    edges = []
    for row_index, row in df.iterrows():
        is_not_nan = ~row.isnull()
        for col_index in row[is_not_nan].index.tolist():
            edges.append((row_index, col_index))
    return edges
