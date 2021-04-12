from typing import List, Dict

import pandas as pd


def load_single_table(
    table_path: str
) -> List[Dict]:
    """
    """
    df = pd.read_csv(table_path, delimiter="\t")
    table_name = table_path.split("/")[-1].rstrip(".tsv")

    facts = []
    for _, row in df.iterrows():
        fact_id = row["[SKIP] UID"]
        chunks = [str(x) for x in row.values[:-4] if not pd.isna(x)]
        fact = " ".join(chunks)
        facts.append({
            "id": fact_id,
            "table": table_name,
            "fact": fact
        })
    return facts


def load_tables_from_dir(tables_dir: str):

    tables_dir = tables_dir.rstrip("/")

    # Get table names
    table_names = []
    with open(f"{tables_dir}/tableindex.txt", 'r') as f:
        for line in f.readlines():
            table_name = line.strip("\n")
            table_names.append(table_name)

    # Load the knowledge
    tables = []
    for name in table_names:
        table = load_single_table(f"{tables_dir}/tables/{name}")
        tables.extend(table)

    return tables
