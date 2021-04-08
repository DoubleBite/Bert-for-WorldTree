import os
import pandas as pd
import networkx as nx

from libs.tools.preprocessing import get_keyword_tokens
from libs.knowledge_graph.utils import load_nodes, load_edges


class WorldTreeKnowledgeGraph(nx.Graph):

    def query(self, query):

        matched_facts = []
        for fact_id, fact_data in self.nodes(data=True):

            query_words = get_keyword_tokens(query)  # A set

            if not "keywords" in fact_data:
                fact_data["keywords"] = get_keyword_tokens(
                    fact_data["description"])
            fact_words = fact_data["keywords"]

            intersection_words = query_words & fact_words
            if intersection_words:
                matched_facts.append(fact_id)
        return matched_facts

    def add_existing_knowledge_graph(self,
                                     another_graph: nx.Graph):
        """
            Merge an existing knowledge graph into the current graph.
        """
        self.add_nodes_from(another_graph.nodes(data=True))
        self.add_edges_from(another_graph.edges(data=True))
        return

    def load_knowledge_graphs_from_dir(self, kg_nodes_dir, kg_edges_dir, verbose=False):
        """
        # The loading flow is designed for worldtree dataset directory structure
        print(loading information)
        """

        num_kg = 0

        # Load all nodes
        for file in os.listdir(kg_nodes_dir):
            if file.endswith('.tsv'):
                id_to_node = load_nodes(
                    f"{kg_nodes_dir}/{file}")
                self.add_nodes_from(id_to_node.items())
                num_kg += 1

        # Load all edges
        for file in os.listdir(kg_edges_dir):
            if file.endswith('.tsv'):
                edges = load_edges(
                    f"{kg_edges_dir}/{file}")
                self.add_edges_from(edges)

        print(f"There are {num_kg} sub-graphs in the KG.")
        print(f"There are {len(self.nodes)} nodes in the KG.")
        print(f"There are {len(self.edges)} edges in the KG.")
        return
