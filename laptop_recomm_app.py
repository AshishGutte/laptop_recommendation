
import streamlit as st
import pandas as pd
import json
import re

from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Custom filter functions ---
from filter import filter_by_price, filter_by_specifications, filter_by_purpose


# --- Styling ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    html, body, [class*="css"]  { font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

st.title("💻 Chat with Laptop Data (Filters + Follow-ups)")


# --- Session State Setup ---
if "main_query_asked" not in st.session_state:
    st.session_state.main_query_asked = False
    st.session_state.main_query_text = ""
if "response_history" not in st.session_state:
    st.session_state.response_history = []

# --- Reset Button ---
if st.button("🔄 Start Over"):
    st.session_state.main_query_asked = False
    st.session_state.main_query_text = ""
    st.session_state.response_history = []

# --- File Upload ---
uploaded_file = st.file_uploader("Upload cleaned Flipkart laptop CSV", type=["csv"])


# --- LangChain Chat History Store ---
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Utility: Laptop-related query check ---
def is_laptop_related(query: str) -> bool:
    keywords = ["laptop", "notebook", "macbook", "chromebook", "gaming", "ultrabook"]
    return any(k in query.lower() for k in keywords)


# --- MAIN LOGIC ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["text"] = (
        df["Brand"] + ", " + df["Product Name"]
        + ", ₹" + df["Price"].astype(str)
        + ", Processor: " + df["Processor"]
        + ", RAM: " + df["RAM"]
        + ", Storage: " + df["Storage"]
        + ", Display: " + df["Display"]
        + ", Specs: " + df["Specifications"]
        + ", URL: " + df["Product URL"]
    )

    # Show past Q&A
    if st.session_state.response_history:
        st.markdown("### 📜 Previous Q&A:")
        for i, entry in enumerate(st.session_state.response_history):
            st.markdown(f"**Q{i+1}:** {entry['query']}")
            st.write(f"**A{i+1}:** {entry['response']}")

    # Main user input
    if not st.session_state.main_query_asked:
        user_input = st.text_input("💬 Ask a question about laptops:")
    else:
        st.markdown(f"💬 **Main Question:** {st.session_state.main_query_text}")
        user_input = None

    if user_input:
        if not is_laptop_related(user_input):
            st.markdown("### 🧠 Answer:")
            st.write("I can only help with laptop-related questions.")
        else:
            st.session_state.main_query_text = user_input
            st.session_state.main_query_asked = True

            # --- FILTERING ---
            filtered_df = filter_by_price(df, user_input)

            if not filtered_df.empty:
                temp_file = "temp_filtered.csv"
                filtered_df[["text"]].to_csv(temp_file, index=False)

                # --- VECTOR DB ---
                loader = CSVLoader(file_path=temp_file)
                documents = loader.load()
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                db = FAISS.from_documents(split_docs, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 5})

                # --- LLM + Prompt ---
                system_prompt = (
                    "You are an expert laptop recommendation assistant. Use the following context:\n\n{context}\n\n"
                    "For each laptop you suggest, include:\n"
                    "- Price\n"
                    "- Key specs (processor, RAM, storage)\n"
                    "- Product URL\n"
                    "- Why it fits user needs (e.g., gaming, student, budget-friendly)"
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])

                llm = Ollama(model="llama3.2")
                retriever_with_history = create_history_aware_retriever(llm, retriever, prompt)
                qa_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever_with_history, qa_chain)

                conversational_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

                # --- Run RAG ---
                session_id = "user-123"
                response = conversational_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )

                st.markdown("### 🧠 Answer:")
                st.write(response["answer"])
                st.session_state.response_history.append({
                    "query": user_input,
                    "response": response["answer"]
                })
            else:
                st.markdown("### 🧠 Answer:")
                st.write("No laptops matched your query after filtering.")


# --- FOLLOW-UP HANDLER ---
def answer_from_history(follow_up_question, history, model="llama3.2"):
    llm = Ollama(model=model)
    history_context = json.dumps(history, indent=2)

    prompt = f"""
You are a highly knowledgeable and helpful laptop recommendation assistant. The user has previously asked about laptops, and you now have access to a JSON-formatted history of those previous Q&A responses.

Use this historical data to fully understand the user's preferences and answer the new follow-up question accurately and comprehensively.

Instructions:
- Use the product names, specifications, and URLs mentioned in the history.
- If the user asks for comparisons, summarize strengths/weaknesses.
- If they ask about a specific model, elaborate on specs and ideal use-case.
- Maintain clarity and structure in your response.

Conversation History:
{history_context}

Follow-up Question:
{follow_up_question}

Answer:"""
    return llm.invoke(prompt)


def extract_product_names(combined_responses: str):
    patterns = [
        r'^\d+\.\s*(.+?)[,:\-\u2013]\s*\u20b9',
        r'^[\u2022\-]\s*(.+?)\s*\u20b9',
        r'^\d+\.\s*(.+?)\s*\u20b9',
        r'^\s*(.+?)\s*[-:\u2013]\s*\u20b9'
    ]
    names = []
    lines = combined_responses.strip().split('\n')
    for line in lines:
        for pattern in patterns:
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                names.append(match.group(1).strip())
                break
    return names


# --- FOLLOW-UP UI ---
if st.session_state.main_query_asked and st.session_state.response_history:
    follow_up = st.text_input("🔁 Ask a follow-up question based on the above answer:")

    if follow_up:
        combined_responses = " ".join([item["response"] for item in st.session_state.response_history])
        possible_names = extract_product_names(combined_responses)
        normalized_names = [name.lower() for name in possible_names]

        numbers_mentioned = re.findall(r'\b(?:number\s*|laptop\s*|about\s*|the\s*)?(\d+)(?:st|nd|rd|th)?\s*laptop?\b', follow_up.lower())
        indices = [int(n) - 1 for n in numbers_mentioned if 0 < int(n) <= len(possible_names)]

        related = bool(indices) or any(name in follow_up.lower() for name in normalized_names)

        if related:
            for i in indices:
                follow_up += f" ({possible_names[i]})"
            follow_up_response = answer_from_history(follow_up, st.session_state.response_history)
            st.markdown("### 🔁 Follow-up Answer:")
            st.write(follow_up_response)
        else:
            st.markdown("### 🔁 Follow-up Answer:")
            st.write("Please ask about a laptop mentioned earlier (e.g., specific model or feature).")
