import pandas as pd
from neo4j import GraphDatabase

# Load CSV
df = pd.read_csv("filtered_tax_cases.csv")

uri = "bolt://localhost:7689"
username = "neo4j"
password = "neo4j123"

driver = GraphDatabase.driver(uri, auth=(username, password))

def build_graph(tx, df):
    for idx, row in df.iterrows():
        case_id = f"Case_{idx}"
        title = row['Titles'].replace('"', "'")

        tx.run(f"""
            MERGE (c:Case {{id: '{case_id}'}})
            SET c.title = "{title}", c.type = "{row['Case_Type']}", c.court = "{row['Court_Name']}"
        """)

        for field in ["Facts", "Issues", "PetArg", "RespArg", "Section", "Precedent", "CDiscource", "Conclusion"]:
            content = str(row[field]).strip().replace('"', "'")
            if content and content.lower() != "nan":
                tx.run(f"""
                    MATCH (c:Case {{id: '{case_id}'}})
                    MERGE (s:Section {{id: '{case_id}_{field}'}})
                    SET s.label = "{field}", s.content = "{content}"
                    MERGE (c)-[:HAS_SECTION {{type: '{field}'}}]->(s)
                """)

with driver.session() as session:
    session.execute_write(build_graph, df)

print("Graph created successfully in Neo4j")