from chromadb import PersistentClient

client = PersistentClient(path="APP/chroma_db")
print("collections:", client.list_collections())
for col in client.list_collections():
    name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
    print(name, "->", client.get_collection(name).count())