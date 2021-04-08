import pandas as pd


def load_knowledge_table(table_path):

    df = pd.read_csv(table_path, delimiter="\t")
    table_name = table_path.split("/")[-1].rstrip(".tsv")

    id_to_knowledge = {}
    for _, row in df.iterrows():
        field_values = [str(x) for x in row.values[:-4] if not pd.isna(x)]
        knowledge_id = row["[SKIP] UID"]
        id_to_knowledge[knowledge_id] = {
            "table": table_name,
            "fact": " ".join(field_values)
        }
    return id_to_knowledge


def load_knowledge_base(kb_path):

    # Load table names
    table_names = []
    with open(f"{kb_path}/tableindex.txt", 'r') as f:
        for line in f.readlines():
            table_name = line.strip("\n")
            table_names.append(table_name)
    print(f"There are {len(table_names)} tables in total.")

    # Load the knowledge
    id_to_knowledge = {}
    for tb_name in table_names:
        table = load_knowledge_table(f"{kb_path}/tables/{tb_name}")
        id_to_knowledge.update(table)
    print(
        f"There are {len(id_to_knowledge)} pieces of facts in the knowledge base.")

    return id_to_knowledge
