import chromadb

#Connect to ChromaDB
chroma_client=chromadb.HttpClient(host='localhost', port='8000')

# Create a collection in ChromaDB
# collection = chroma_client.create_collection(name="pdfChatbot1")

# Get the collection from ChromaDB
collection = chroma_client.get_collection(name='pdfChatbot1')

# Delete the collection from ChromaDB
# chroma_client.delete_collection(name='my_collection')


# List down all the collections in ChromaDB
# collections = chroma_client.list_collections()
# for collection in collections:
#     print(f"collection name is : {collection.name}")

# Query the collection from ChromaDB
results = collection.query(
    query_texts=["Harry Potter school name"],
    n_results=1
)
# print(results)
documents = results['documents']
# print(documents)
# Iterate over the documents
for document in documents:
    print(document[0].replace("page_content='", "").split("metadata=")[0])
    # print(document[1].replace("page_content='", "").split("metadata=")[0])
    
