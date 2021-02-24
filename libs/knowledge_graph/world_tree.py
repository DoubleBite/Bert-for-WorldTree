import os
import pandas as pd
import networkx as nx

from libs.tools.preprocessing import get_keyword_tokens


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

    def load_knowledge_graphs_from_dir(self, kg_dir, verbose=False):
        """
        print(loading information)
        """
        # The loading flow is designed for worldtree dataset directory structure
        all_id_to_node = {}
        for file in os.listdir(kg_dir):
            if file.endswith('.tsv'):
                id_to_node = self.load_knowledge_graph_from_path(
                    f"{kg_dir}/{file}")
                all_id_to_node.update(id_to_node)

        print(len(all_id_to_node))

        return all_id_to_node
