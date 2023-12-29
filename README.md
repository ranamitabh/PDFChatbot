# PDFChatbot

Setup chroma db

docker pull chromadb/chroma

docker run -p 8000:8000 chromadb/chroma


# Codes under "BasicChromaDBOperation" shows the chromadb basic functions like create, delete, write and read functions. It reads a file, tokenize it, store it into a chromadb (Vector Database) and then query it. it uses all-MiniLM-L6-v2.

# "ChatBotlangchain" has codes which uses Facebook AI Similarity Search (FAISS). 
