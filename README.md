# 🚀 FocusED: A RAG Productivity Hub


FocusED is an all-in-one, local-first AI application designed to enhance your study and work sessions. Built with Streamlit and powered by LangChain, it combines a structured work timer with a powerful AI assistant that can understand and process your PDF documents.

This tool allows you to upload a document, start a timed work session, manage your tasks, and interact with an AI that acts as an expert on your uploaded content, helping you learn faster and more efficiently.

---

## Table of Contents
- [Features](#-features)
- [Architecture & Core Technologies](#️-architecture--core-technologies)
- [Setup and Installation](#️-setup-and-installation)
- [How to Run the Application](#️-how-to-run-the-application)
- [Contributing](#-contributing)

---

## ✨ Features

* **Structured Work Sessions:** A built-in 90-minute work timer followed by a mandatory 30-minute rest period to promote focused, healthy productivity.
* **Persistent To-Do List:** Manage your tasks directly in the sidebar. Your list is automatically saved to a `tasks.json` file, so your progress is never lost between sessions.
* **Retrieval-Augmented Generation (RAG):** Ask specific questions about your uploaded PDF and get answers that are grounded in the document's content.
* **Retrieved Source Verification:** See the exact text chunks from the document that the AI used to generate its answer, ensuring accuracy and trust.
* **General AI Actions:** Perform high-level tasks on the entire document with a single click:
    * **Summarize:** Get a concise summary of the whole document.
    * **Simplify:** Have complex concepts explained in simple terms.
    * **Create Flashcards:** Automatically generate question/answer pairs for studying.
    * **Generate Quiz:** Create a multiple-choice quiz to test your knowledge.
* **Conversational Chat History:** Your entire interaction with the AI is displayed in a clean, chat-like interface.

---

## 🛠️ Architecture & Core Technologies

The application is built on a modern, local-first AI stack:

* **Frontend:** **Streamlit** is used to create the entire interactive user interface with pure Python.
* **Orchestration:** **LangChain** connects all the different components, from document loading to the final answer generation.
* **LLM:** **Hugging Face `transformers`** runs the `google/flan-t5-base` model locally on your machine for all text generation tasks.
* **Text Extraction:** **PyMuPDF (`fitz`)** efficiently extracts all text content from uploaded PDF files.
* **Vector Store:** **FAISS** (Facebook AI Similarity Search) creates a high-speed, in-memory vector database for efficient information retrieval.
* **Embeddings:** **SentenceTransformers** (`all-MiniLM-L6-v2`) converts text chunks into numerical vectors that capture their semantic meaning.

---

## ⚙️ Setup and Installation

Follow these steps to get FocusED running on your local machine.

### 1. Prerequisites

* **Python 3.8+:** Ensure you have a modern version of Python installed.
* **Hugging Face Account:** You will need a free Hugging Face account to get an API token.

### 2. Create Your Project Files

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/FocusED.git](https://github.com/your-username/FocusED.git)
    cd FocusED
    ```

2.  **Create the `.env` File:** In the project's root directory, create a file named `.env`. Get your API token from your [Hugging Face account settings](https://huggingface.co/settings/tokens) and add it to the file like this:
    ```
    HF_API_KEY="hf_YOUR_TOKEN_HERE"
    ```

3.  **Verify the `requirements.txt` File:** The repository includes a `requirements.txt` file with all necessary dependencies.

### 3. Install Dependencies

Open your terminal or command prompt, navigate to your project folder, and run the following command to install all the necessary libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Application

Once the setup is complete, you can start the application with a single command.

1.  **Navigate to your project folder** in the terminal.
2.  **Run the Streamlit command:**
    ```bash
    streamlit run app.py
    ```
Your web browser will automatically open a new tab with the FocusED application running.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features or improvements, please open an issue to discuss what you would like to change.

Please make sure to update tests as appropriate.

