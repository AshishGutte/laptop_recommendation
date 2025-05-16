import streamlit as st
import pandas as pd
import json
import re
import os 

from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Styling ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    html, body, [class*="css"]  { font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

st.title("💻 Chat with Laptop Data")

# --- Session State Setup ---
if "main_query_asked" not in st.session_state:
    st.session_state.main_query_asked = False
    st.session_state.main_query_text = ""
if "response_history" not in st.session_state:
    st.session_state.response_history = []
if "conversational_rag_chain" not in st.session_state: # To store the main chain
    st.session_state.conversational_rag_chain = None
if "uploaded_file_name_cache" not in st.session_state: # To help with re-initialization logic
    st.session_state.uploaded_file_name_cache = None

# --- LangChain Chat History Store (Modified for Streamlit persistence) ---
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_history_store[session_id]

# --- Reset Button ---
if st.button("🔄 Start Over"):
    st.session_state.main_query_asked = False
    st.session_state.main_query_text = ""
    st.session_state.response_history = []
    st.session_state.conversational_rag_chain = None
    st.session_state.chat_history_store = {} # Clear chat history store as well
    st.session_state.uploaded_file_name_cache = None
    # Potentially clear FAISS index cache if you implement it
    st.rerun()


# --- File Upload ---
uploaded_file = st.file_uploader("Upload cleaned Flipkart laptop CSV", type=["csv"])

# --- Utility: Laptop-related query check ---
def is_laptop_related(query: str) -> bool:
    keywords = ["laptop", "notebook", "macbook", "chromebook", "gaming", "ultrabook"] # Added common terms
    return any(k in query.lower() for k in keywords)

# --- MAIN LOGIC ---
if uploaded_file:
    # Basic check to see if file changed, to re-initialize chain if necessary
    if st.session_state.uploaded_file_name_cache != uploaded_file.name:
        st.session_state.conversational_rag_chain = None # Reset chain if new file
        st.session_state.chat_history_store = {} # Reset history for new file context
        st.session_state.response_history = [] # Reset display history
        st.session_state.main_query_asked = False
        st.session_state.uploaded_file_name_cache = uploaded_file.name


    temp_csv_path = "uploaded_laptops_temp.csv"
    try:
        df = pd.read_csv(uploaded_file)
        df.to_csv(temp_csv_path, index=False)
        loader = CSVLoader(file_path=temp_csv_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        st.stop()


    # Show past Q&A
    if st.session_state.response_history:
        st.markdown("### 📜 Previous Q&A:")
        for i, entry in enumerate(st.session_state.response_history):
            st.markdown(f"**Q{i+1}:** {entry['query']}")
            st.write(f"**A{i+1}:** {entry['response']}")

    # User input
    if not st.session_state.main_query_asked:
        user_input = st.text_input("💬 Ask a question about laptops:", key="main_input")
    else:
        st.markdown(f"💬 **Main Question:** {st.session_state.main_query_text}")
        user_input = None 

    if user_input: 
        if not is_laptop_related(user_input):
            st.markdown("### 🧠 Answer:")
            st.write("I can only help with laptop-related questions for your initial query.")
        else:
            st.session_state.main_query_text = user_input
            st.session_state.main_query_asked = True

            with st.spinner("Processing your request... 🤔"):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 10}) 

                llm = Ollama(model="llama3.2") 

                # 1. Prompt for History-Aware Retriever (Question Rephrasing)
                contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                qa_system_prompt = """
                You are an expert assistant for laptop recommendations, specializing in matching user queries to laptops based on price, specifications, and use cases.

                Use the following retrieved context to answer the question. If you don't know the answer, say that you don't know.
                Each entry in the context may include: Product Name, Rating, Price, Product URL, Battery Life, Weight, Webcam, Display Size, Brand, Processor, RAM, DDR, OS, Storage, Display, Warranty.

                Context:\n\n{context}\n\n

                Your job is to recommend laptops based on the user's query. Structure your response with:

                - **Brand**
                - **Product Name**
                - **Price** (ensure currency if available, e.g., ₹)
                - **RAM**
                - **Product URL**
                - **Key Specifications** (summarize relevant specs from the context directly pertaining to the query and general importance)
                - **Why this is recommended**: Elaborate clearly on why this specific laptop is an excellent match for the user's query. Connect its specific features (from the context) directly to their stated needs or use case (from the query or chat history). Highlight its key strengths and benefits relevant to their request, explaining *why* it's a superior choice for them.

                Guidelines:
                - ONLY use the information present in the uploaded data (retrieved context). Do not make up information or refer to external knowledge. Assume the provided context is the source of truth for your response.
                - For price range queries (e.g., "50000 to 70000"), use the 'Price' field from the context.
                - For feature queries (e.g., "16GB RAM", "512GB SSD", "Intel Core i5"), use relevant fields like RAM, Storage, Processor from the context.
                - Use case matching (refer to context for specs):
                    - Software engineer: Suggest laptops with generally ≥16GB RAM, ≥512GB SSD, modern Intel i5/i7 or Ryzen 5/7, and reasonable weight (e.g., <2kg). Justify based on these specs.
                    - Gaming: Suggest laptops with generally ≥16GB RAM, ≥512GB SSD, powerful Intel i7/i9 or Ryzen 7/9, and a dedicated GPU if mentioned in context. Justify based on these specs.
                    - Student: Suggest laptops with generally ≥8GB RAM, ≥256GB SSD, efficient Intel i3/i5 or Ryzen 3/5, and consider price constraints if mentioned (e.g., < ₹40000). Justify based on these specs.
                - Use chat history to understand context for follow-up questions like “what about the second one you mentioned?”, “compare it with the first one”, or if the user refers to “same budget”.
                - Recommend up to 3 laptops if multiple good matches are found, preferably sorted by price or relevance.
                - If no suitable laptops are found in the context matching the criteria: "Sorry, no laptops found in the provided data matching your criteria."
                - Be concise, structured, and helpful.
                - **Do NOT** include any disclaimers stating that specifications are "based on the information provided in the context and may not be exhaustive or up-to-date."
                - **Do NOT** end your response with generic inviting phrases like "If you'd like to know more about this laptop or compare it with other options, feel free to ask!". Answer the user's current query comprehensively and then conclude.
                """
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

                rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

                st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                session_id = "user_session_123" # Use a more robust session ID management if needed
                response = st.session_state.conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response.get("answer", "Sorry, I couldn't generate a response.")

                st.markdown("### 🧠 Answer:")
                st.write(answer)
                st.session_state.response_history.append({
                    "query": user_input,
                    "response": answer
                })
                st.rerun() # Rerun to update UI, clear main input, and show follow-up input

# --- FOLLOW-UP UI ---
if st.session_state.main_query_asked and uploaded_file: # Ensure a main query has been asked and file is still loaded
    follow_up_input = st.text_input("🔁 Ask a follow-up question based on the above answer(s):", key="follow_up_input")

    if follow_up_input:
        if st.session_state.conversational_rag_chain:
            with st.spinner("Thinking about your follow-up... 🤔"):
                session_id = "user_session_123" # Must match the session_id used before
                response = st.session_state.conversational_rag_chain.invoke(
                    {"input": follow_up_input},
                    config={"configurable": {"session_id": session_id}}
                )
                follow_up_answer = response.get("answer", "Sorry, I couldn't process that follow-up.")

                st.markdown("### 🔁 Follow-up Answer:")
                st.write(follow_up_answer)
                st.session_state.response_history.append({
                    "query": follow_up_input,
                    "response": follow_up_answer
                })

        else:
            st.warning("The conversational chain is not initialized. Please ask a main question first or try starting over.")

elif not uploaded_file:
    st.info("Please upload a CSV file to begin.")