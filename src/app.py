import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import streamlit as st

# Load Sentence-Transformer model for encoding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load T5 model for summarization
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")
summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Function to encode texts into embeddings
def encode_texts(texts):
    return embedding_model.encode(texts)

# Function to summarize text
def summarize(text):
    inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Initialize FAISS index for text retrieval
def initialize_faiss(corpus):
    embeddings = encode_texts(corpus)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Function to perform retrieval from FAISS index
def retrieve_relevant_documents(query, index, corpus, top_k=3):
    query_embedding = encode_texts([query])[0].reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [corpus[i] for i in indices[0]]
    return retrieved_docs

# Sample contract corpus (this can be replaced with your real corpus)
corpus = [
    "This agreement is made between the parties for the sale of goods.",
    "The buyer agrees to pay the seller a total amount of $10,000.",
    "The seller will deliver the goods within 30 days of receiving payment.",
    "Any disputes arising from this agreement will be resolved by arbitration.",
    "This contract is governed by the laws of the state of California."
]

# Initialize FAISS index with the corpus
index = initialize_faiss(corpus)

# Streamlit Web App
def main():
    st.title("Contract Review Assistant")
    
    # User input for contract query
    query = st.text_area("Enter your contract-related query:", "What are the payment terms?")
    
    if st.button('Get Relevant Information'):
        if query:
            # Step 1: Retrieve relevant documents
            retrieved_docs = retrieve_relevant_documents(query, index, corpus)
            st.subheader("Relevant Information Retrieved:")
            for doc in retrieved_docs:
                st.write(doc)
            
            # Step 2: Generate summary from the retrieved documents
            combined_text = " ".join(retrieved_docs)
            summary = summarize(combined_text)
            
            st.subheader("Generated Summary:")
            st.write(summary)
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()
