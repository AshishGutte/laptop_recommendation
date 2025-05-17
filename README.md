---

# 💻 Conversational Laptop Recommendation System

This project is a conversational chatbot that recommends laptops based on user queries. It uses **LangChain**, **FAISS**, and **Ollama (LLaMA 3.2)** for Retrieval-Augmented Generation (RAG). A Streamlit interface enables users to interact naturally, ask follow-up questions, and receive detailed recommendations.

---

## 🧠 What It Does

* Understands user queries like “Suggest laptops under ₹60000 for programming”
* Retrieves relevant laptops from a local dataset
* Uses a local LLM to generate personalized responses
* Remembers previous interactions for follow-up queries
* Returns laptop specs and product links

---

## 🗂️ Project Workflow Overview

1. **Web Scraping** (external process)
2. **Data Cleaning** (`data_cleaning.py`)
3. **RAG-based Chatbot App** (`laptop_recommendation_app.py`)
4. **User Interaction via Streamlit**

---

## 🌐 Step 1: Web Scraping

* Laptop listings are scraped from Flipkart using automation tools like **Selenium** or **BeautifulSoup**.
* Fields like product name, price, and descriptions are extracted.
* The result is saved as a raw CSV file containing mixed and inconsistent data.

> **Note:** Web scraping is performed before running the main app. This step is not automated in the app and needs to be done manually or via an external script.

---

## 🧼 Step 2: Data Cleaning

* The raw scraped CSV file is processed using the `data_cleaning.py` script.
* Key tasks include:

  * Cleaning the price field and removing unwanted characters
  * Standardizing product names
  * Extracting specs such as Processor, RAM, Storage, OS, Display, and Warranty using regular expressions
* The cleaned data is saved into a new CSV file that is used by the chatbot app

---

## 💬 Step 3: Chatbot Recommendation App

* Launch the Streamlit app (`laptop_recommendation_app.py`)
* Upload the cleaned CSV file
* Ask a natural question, such as “Recommend laptops for gaming under ₹70000”
* The app will:

  * Convert laptop rows into searchable documents
  * Use **FAISS** for semantic similarity search
  * Run a **LangChain QA chain** with **Ollama** for generating a response
  * Display results with laptop names, specs, prices, and links

---

## 🔁 Step 4: Follow-Up Questions

* The app keeps memory of previous queries
* You can ask context-aware questions like:

  * “What about lighter options?”
  * “Compare the first two laptops”
* The system uses chat history to tailor each answer accordingly

---

## 🧠 How It Works Internally

* **LangChain** manages the RAG pipeline and memory
* **FAISS** searches top relevant laptops from the cleaned dataset
* **Ollama** provides fast, local inference with **LLaMA 3.2**
* **Streamlit** handles the interactive chat interface
* The app formats answers as structured JSON to present clean outputs with specs and Flipkart URLs

---

## 🎯 Example Use Cases

* Students seeking budget laptops for studying
* Engineers needing high-performance devices for coding
* Gamers asking for laptops with strong GPUs
* Users comparing models before purchasing

---

## 🧪 Tech Stack

| Purpose         | Tool/Library          |
| --------------- | --------------------- |
| UI              | Streamlit             |
| LLM             | Ollama (LLaMA 3.2)    |
| Embeddings      | Sentence Transformers |
| Vector Search   | FAISS                 |
| Prompt Handling | LangChain             |
| Data Processing | Pandas, Regex         |
| Data Source     | Flipkart (scraped)    |

---

## ✅ Final Notes

* All responses are based on the uploaded dataset (not live data)
* The app is fully local and doesn’t require internet after setup
* Suitable for offline, private, and personalized recommendation use cases
* Memory allows the chatbot to behave more naturally in a conversation











