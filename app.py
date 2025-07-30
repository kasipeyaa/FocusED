import os
import time
import streamlit as st
import fitz # PyMuPDF
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #Facebook AI Similarity Search
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY") # This is loaded but not explicitly used by the local pipeline

# --- App State Management ---
if 'timer_status' not in st.session_state:
    st.session_state.timer_status = 'stopped' # States: stopped, running, resting, finished
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
# Add state for the to-do list
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# Setup LLM using a local pipeline
@st.cache_resource
def load_llm():
    """Loads the Hugging Face pipeline for text generation."""
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=512
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# Prompt Template
qa_prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = """
    You are a helpful assistant. Use the context below to answer the user's questions.
    Context: {context}
    Question: {question}
    Answer:
    """
)

# PDF loader using PyMuPDF (fitz)
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.set_page_config(page_title="FocusED", layout="centered")
st.title("📄 FocusED")


# --- Timer Function ---
WORK_DURATION = 90 * 60
REST_DURATION = 30 * 60

def update_timer():
    # This function now populates the placeholder created in the sidebar
    if st.session_state.timer_status == 'running':
        elapsed_time = time.time() - st.session_state.start_time
        remaining_time = max(0, WORK_DURATION - elapsed_time)
        if remaining_time > 0:
            progress_val = elapsed_time / WORK_DURATION
            mins, secs = divmod(int(remaining_time), 60)
            hours, mins = divmod(mins, 60)
            with timer_placeholder.container():
                st.info(f"Work Session: {hours:02d}:{mins:02d}:{secs:02d} Remaining")
                st.progress(progress_val)
        else:
            st.session_state.timer_status = 'resting'
            st.session_state.start_time = time.time()
            st.rerun()

    elif st.session_state.timer_status == 'resting':
        elapsed_time = time.time() - st.session_state.start_time
        remaining_time = max(0, REST_DURATION - elapsed_time)
        if remaining_time > 0:
            progress_val = elapsed_time / REST_DURATION
            mins, secs = divmod(int(remaining_time), 60)
            with timer_placeholder.container():
                st.balloons()
                st.warning(f"**REST PERIOD**\n\nTime left: {mins:02d}:{secs:02d}")
                st.progress(progress_val)
        else:
            st.session_state.timer_status = 'finished'
            st.rerun()
            
    elif st.session_state.timer_status == 'stopped':
         with timer_placeholder.container():
            if st.button("Start 90-Minute Session"):
                st.session_state.timer_status = 'running'
                st.session_state.start_time = time.time()
                st.rerun()
    
    elif st.session_state.timer_status == 'finished':
         with timer_placeholder.container():
            st.success("🎉 Session Complete!")
            if st.button("Reset Session"):
                st.session_state.timer_status = 'stopped'
                st.session_state.start_time = 0
                st.rerun()

# --- Sidebar ---
st.sidebar.header("⏱️ Session Timer")
timer_placeholder = st.sidebar.empty()
update_timer() # Initial call to display the timer in the sidebar

st.sidebar.header("✅ To-Do List")

def add_new_task():
    """Callback function to add a new task."""
    task_description = st.session_state.new_task_input
    if task_description:
        st.session_state.tasks.append({'task': task_description, 'completed': False})
        # Clear the input box after adding
        st.session_state.new_task_input = ""

def update_task_status(task_index):
    """Callback to update the completion status of a task."""
    # The checkbox's new value is stored in session_state under its key.
    # We use this value to update our canonical list of tasks.
    st.session_state.tasks[task_index]['completed'] = st.session_state[f"task_{task_index}"]

st.sidebar.text_input(
    "Add a new task:", 
    key="new_task_input", 
    on_change=add_new_task,
    placeholder="Type a task and press Enter"
)

# Display the list of tasks
if 'tasks' in st.session_state and st.session_state.tasks:
    for i, task_item in enumerate(st.session_state.tasks):
        st.sidebar.checkbox(
            label=task_item['task'], 
            value=task_item['completed'], 
            key=f"task_{i}",
            on_change=update_task_status,
            args=(i,) # Pass the task's index to the callback
        )
else:
    st.sidebar.write("No tasks yet.")


# --- Main App Body (disabled during rest) ---
if st.session_state.timer_status != 'resting':
    st.markdown("""
    Upload a PDF document and ask questions about its content.
    """)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Check if the file has already been processed to avoid reprocessing on every rerun
        if st.session_state.get('processed_file_name') != uploaded_file.name:
            with st.spinner("Hold on I'm processing the File... This might take a moment."):
                raw_text = extract_text_from_pdf(uploaded_file)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(raw_text)
                embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectordb = FAISS.from_texts(chunks, embedding=embedder)
                retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                
                qa_chain = RetrievalQA.from_chain_type (
                    llm = llm,
                    retriever = retriever, 
                    chain_type = "stuff",
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": qa_prompt}
                )
                
                st.success("I'm done! You can ask questions now^^")
                st.session_state.qa_chain = qa_chain
                st.session_state.processed_file_name = uploaded_file.name

    # Chat Interface
    if "qa_chain" in st.session_state:
        st.header("💬 Ask a Question")
        question = st.text_input("Enter your question about the file.")

        if question:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.qa_chain.invoke(question)
                    st.subheader("Answer")
                    st.write(result.get('result', 'No answer could be generated.'))
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {str(e)}")
else:
    # This message is shown in the main area during the rest period
    st.info("You're resting girl. Just chill for a sec💀")

# Auto-refresh loop for the timer
if 'timer_status' in st.session_state and st.session_state.timer_status in ['running', 'resting']:
    time.sleep(1)
    st.rerun()