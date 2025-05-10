import streamlit as st
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from filter import filter_by_price, filter_by_specifications, filter_by_purpose

st.title("Chat with Laptop Data 💻 (with Price Filters + URLs)")

uploaded_file = st.file_uploader("flipkart_laptop_cleaned.csv", type=["csv"])
user_input = st.text_input("Ask a question about laptops:")

# Chat session memory
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def parse_comparison_query(query, laptops_df):
    """Parse user query to identify laptops for comparison or specifications."""
    # Basic parsing logic (you can extend this based on your needs)
    query = query.lower()
    if "compare" in query:
        # Look for numbers like "compare 2 and 3"
        numbers = [int(s) for s in query.split() if s.isdigit()]
        if len(numbers) == 2:
            laptop_1 = laptops_df.iloc[numbers[0] - 1]  # Adjust for 0-based index
            laptop_2 = laptops_df.iloc[numbers[1] - 1]
            return f"Comparing Laptop {numbers[0]} and Laptop {numbers[1]}:\n\n" + \
                   f"**Laptop {numbers[0]}:** {laptop_1['Brand']} - {laptop_1['Product Name']} - ₹{laptop_1['Price']}\n" + \
                   f"**Laptop {numbers[1]}:** {laptop_2['Brand']} - {laptop_2['Product Name']} - ₹{laptop_2['Price']}"
    elif "specifications" in query:
        # Look for numbers like "laptop 1 specifications"
        numbers = [int(s) for s in query.split() if s.isdigit()]
        if numbers:
            laptop = laptops_df.iloc[numbers[0] - 1]  # Adjust for 0-based index
            return f"Specifications of Laptop {numbers[0]}:\n" + \
                   f"**Brand:** {laptop['Brand']}\n" + \
                   f"**Product Name:** {laptop['Product Name']}\n" + \
                   f"**Price:** ₹{laptop['Price']}\n" + \
                   f"**Processor:** {laptop['Processor']}\n" + \
                   f"**RAM:** {laptop['RAM']}\n" + \
                   f"**Storage:** {laptop['Storage']}\n" + \
                   f"**Display:** {laptop['Display']}\n" + \
                   f"**Specifications:** {laptop['Specifications']}\n" + \
                   f"**Product URL:** {laptop['Product URL']}"
    return "Sorry, I didn't understand your query. Try asking something like 'compare 1 and 2' or 'specifications of laptop 1'."

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    # Prepare data for search
    df["text"] = (
        df["Brand"].astype(str)
        + ", " + df["Product Name"].astype(str)
        + ", ₹" + df["Price"].astype(str)
        + ", Processor: " + df["Processor"].astype(str)
        + ", RAM: " + df["RAM"].astype(str)
        + ", Storage: " + df["Storage"].astype(str)
        + ", Display: " + df["Display"].astype(str)
        + ", Specs: " + df["Specifications"].astype(str)
        + ", URL: " + df["Product URL"].astype(str)
    )

    # Apply filters based on user input
    filtered_df = filter_by_price(df, user_input)
    filtered_df = filter_by_specifications(filtered_df, user_input)
    filtered_df = filter_by_purpose(filtered_df, user_input)

    # Check if filtered dataframe is not empty
    if not filtered_df.empty:
        # Save filtered data
        temp_filtered_file = "temp_filtered.csv"
        filtered_df[["text"]].to_csv(temp_filtered_file, index=False)

        loader = CSVLoader(file_path=temp_filtered_file)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.from_documents(split_docs, embeddings)
        retriever = db.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest user question, reformulate the question to be a standalone query if needed."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        llm = Ollama(model="llama3.2")
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an expert laptop recommendation assistant. Use the following context:\n\n{context}\n\n"
            "For each laptop you suggest, include:\n"
            "- Price\n"
            "- Key specs (e.g., processor, RAM, storage)\n"
            "- Product URL\n"
            "- A **brief reason** why this laptop fits the user's needs (e.g., 'great for gaming', 'ideal for students', 'lightweight for travel', 'powerful CPU for multitasking').\n\n"
            "Be concise and helpful."
        )


        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # 🔥 Invoke RAG for user question
        if user_input:
            session_id = "user-123"
            response = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            # Display initial response
            st.markdown("### 🤖 Answer:")
            st.write(response["answer"])

            # Extract recommended laptops from the response text
            recommended_laptops = filtered_df.head(5).reset_index()  # You can adjust this as needed

            # Display recommended laptops to the user based on the filtered results
            st.markdown("### 🔍 Recommended Laptops:")
            for idx, row in recommended_laptops.iterrows():
                st.markdown(f"**{idx + 1}. {row['Brand']} - {row['Product Name']}** — ₹{row['Price']}")

            # Chatbot-like input for further assistance
            query_for_details = st.text_input("Ask about the recommended laptops (e.g., compare 1 and 2 or specifications of laptop 3 and also tell for each laptop why this laptop is best from others):")

            if query_for_details:
                details_response = parse_comparison_query(query_for_details, recommended_laptops)
                st.markdown("### 🤖 Additional Information:")
                st.write(details_response)

    else:
        st.warning("No laptops found matching your criteria.")




