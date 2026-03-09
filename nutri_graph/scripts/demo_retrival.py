import duckdb
from nutri_graph.kb.search import search_food

if __name__ == "__main__":
    con = duckdb.connect("data/nutri_kb.duckdb", read_only=True)
    results = search_food(con, "milk", k=5)
    print(results)