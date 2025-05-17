Thanks! You’ve added great information, especially about `laptop_recommendation_app.py`. I’ll now revise your `README.md` to improve:

* ✅ Flow and structure
* ✅ Remove duplicate sections
* ✅ Fix formatting issues (e.g., escaped backslashes, code block closures)
* ✅ Ensure consistency and clarity

---

### ✅ Final Cleaned and Structured `README.md`

```markdown
# 💻 Conversational Laptop Recommendation System (RAG-based with LangChain)

This is an intelligent chatbot system that provides personalized laptop recommendations based on user queries. It uses RAG (Retrieval-Augmented Generation) with LangChain, FAISS for vector search, and Ollama for local LLM inference (LLaMA 3.2). The app supports follow-up questions using memory and works on cleaned Flipkart laptop data.

---

## 🚀 Features

- ✅ Conversational UI built with **Streamlit**
- ✅ Supports **follow-up questions** with memory
- ✅ Embedding-powered semantic search using **FAISS**
- ✅ Uses **LangChain** with local LLM via **Ollama (LLaMA 3.2)**
- ✅ Real-time laptop recommendations using specs and price
- ✅ Parses and cleans Flipkart laptop data
- ✅ Recommends based on user needs (e.g., gaming, student, engineer)

---

## 📁 Project Structure

```

├── data\_cleaning.py               # Script to clean Flipkart laptop data
├── laptop\_recommendation\_app.py  # Main Streamlit chatbot app
├── flipkart\_laptop\_final.csv     # Raw data (from web scraping)
├── flipkart\_laptop\_cleann.csv    # Cleaned dataset
├── README.md                      # This file

````

---

## 🔍 Inside `laptop_recommendation_app.py`

This is the **heart of the project**, combining LangChain, FAISS, Ollama, and Streamlit into a conversational app.

### 🔧 Key Modules:

| Function / Component      | Description |
|---------------------------|-------------|
| `load_data`               | Loads and parses the cleaned CSV into a Pandas DataFrame |
| `generate_docs_from_csv`  | Converts each laptop row into a LangChain Document |
| `create_vectorstore`      | Embeds documents using `sentence-transformers` and builds a FAISS index |
| `get_top_n_documents`     | Retrieves top 10 documents for a user query using similarity search |
| `format_docs`             | Converts LangChain docs into clean string format for LLM |
| `setup_qa_chain`          | Creates a LangChain RetrievalQA chain using Ollama |
| `run_conversational_qa`   | Handles both new and follow-up queries using chat history |
| `build_json_response`     | Extracts structured JSON response with specs and URLs |
| `build_chat_interface`    | Streamlit UI: handles file upload, input, and chat history display |

---

## 🧼 Step 1: Clean the Data

The `data_cleaning.py` script:

- Extracts and cleans the `Price` field
- Standardizes the `Product Name`
- Extracts specs like `Processor`, `RAM`, `DDR`, `OS`, `Storage`, `Display`, and `Warranty` using regex
- Saves the cleaned file as `flipkart_laptop_cleann.csv`

### 📦 Run the cleaning script

```bash
python data_cleaning.py


---

## 💬 Step 2: Run the Chatbot App

Make sure the cleaned file (`flipkart_laptop_cleann.csv`) is available.

### 1. 🔧 Set up your environment

```bash
# Create and activate a virtual environment
python -m venv real
source real/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### 2. 🦙 Install Ollama and LLaMA 3.2

Make sure [Ollama](https://ollama.com/) is installed and the `llama3.2` model is downloaded:

```bash
ollama run llama3.2
```

### 3. 🚀 Run the Streamlit app

```bash
streamlit run laptop_recommendation_app.py
```

---

## 🧠 How It Works

1. **CSV Upload**: Upload your cleaned Flipkart laptop CSV.
2. **Initial Query**: Ask a question like *"Suggest laptops under ₹60000 for programming"*.
3. **RAG Pipeline**:

   * Embeds the laptop data and performs semantic retrieval
   * Uses Ollama (LLaMA 3.2) to generate answers based on top retrieved docs
   * Formats the response into a JSON structure (name, price, URL, etc.)
   * Stores previous responses to support follow-up questions
4. **Follow-up**: Ask questions like *"What about gaming ones?"* or *"Compare the first two."*

---

## 🧪 Tech Stack

| Component       | Tool/Library                   |
| --------------- | ------------------------------ |
| UI              | Streamlit                      |
| LLM             | Ollama (`llama3.2`)            |
| Vector Search   | FAISS                          |
| Embeddings      | Sentence-Transformers (MiniLM) |
| Memory/Chaining | LangChain                      |
| Data Handling   | Pandas                         |
| Data Source     | Flipkart (via scraping)        |

---

## 📌 Sample Queries

* *"Suggest laptops for a software engineer under ₹80000"*
* *"Give me lightweight laptops with 16GB RAM"*
* *"Compare the second one with the first one"*

---

## 📎 Notes

* This app only works on laptop-related queries.
* All answers are generated using only the uploaded dataset context.
* Memory allows follow-up questions referencing previous results.

---








