import streamlit as st
import json
from dotenv import load_dotenv
from agents import ChatbotAgent, SummaryAgent, QuizAgent, FlashcardAgent
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import tempfile
import os
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_chain" not in st.session_state:
    st.session_state.doc_chain = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "doc_chain_dict" not in st.session_state:
    st.session_state.doc_chain_dict = None
if "all_text" not in st.session_state:
    st.session_state.all_text = []

# Page config
st.set_page_config(
    page_title="AI-Powered Learning Assistant",
    page_icon="📚",
    layout="wide"
)

# Streamlit app layout
st.title("📚 AI-Powered Learning Assistant")

# Sidebar to input Groq API Key
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

# Check for the API key
if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

# Initialize agents with error handling
try:
    chatbot_agent = ChatbotAgent(groq_api_key)
    summary_agent = SummaryAgent(groq_api_key)
    quiz_agent = QuizAgent(groq_api_key)
    flashcard_agent = FlashcardAgent(groq_api_key)
except Exception as e:
    st.error(f"Error initializing agents: {str(e)}")
    st.stop()

# Sidebar layout for topic and document upload
with st.sidebar:
    st.markdown("### 📝 Topic & Documents")
    
    # Topic input
    topic = st.text_input(
        "Enter your learning topic:",
        value=st.session_state.current_topic if st.session_state.current_topic else "",
        key="topic_input"
    )
    
    # Update current topic in session state
    if topic != st.session_state.current_topic:
        st.session_state.current_topic = topic
    
    # Document upload
    st.markdown("### 📄 Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload your study materials to get contextual answers"
    )
    
    # Process uploaded documents
    if uploaded_files:
        with st.spinner("Processing documents..."):
            newly_processed = False
            for file in uploaded_files:
                if file.name not in st.session_state.processed_files:
                    try:
                        if file.type == "application/pdf":
                            pdf_reader = PdfReader(file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                            st.session_state.all_text.append(text)
                        else:  # txt files
                            text = file.read().decode()
                            st.session_state.all_text.append(text)
                        
                        st.session_state.processed_files.add(file.name)
                        newly_processed = True
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        continue
            
            if newly_processed and st.session_state.all_text:
                try:
                    st.session_state.doc_chain_dict = chatbot_agent.init_doc_chain("\n".join(st.session_state.all_text))
                    st.success("✅ Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error initializing document chain: {str(e)}")
    
    # Clear documents button
    if st.session_state.processed_files:
        if st.button("Clear Uploaded Documents"):
            st.session_state.processed_files = set()
            st.session_state.doc_chain_dict = None
            st.session_state.all_text = []
            st.experimental_rerun()

# Feature selection
option = st.sidebar.selectbox(
    "Select a Feature",
    ["Chatbot", "Study Material", "Quiz", "Flashcards"]
)

# Check if topic is provided
if not st.session_state.current_topic:
    st.warning("Please enter a topic in the sidebar to begin.")
    st.stop()

# Chatbot for Q&A
if option == "Chatbot":
    st.subheader("💭 Interactive Learning Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask any question about " + st.session_state.current_topic + "..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.doc_chain_dict:
                        response = chatbot_agent.chat_with_docs(
                            prompt,
                            st.session_state.doc_chain_dict,
                            st.session_state.messages
                        )
                    else:
                        response = chatbot_agent.chat(prompt, st.session_state.messages)
                    
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Study Material Generation
elif option == "Study Material":
    st.subheader(f"📖 Study Notes: {st.session_state.current_topic}")
    
    if st.button("Generate Study Material", type="primary"):
        with st.spinner("Generating study material..."):
            try:
                summary = summary_agent.run(st.session_state.current_topic)
                st.markdown(summary)
            except Exception as e:
                st.error(f"Error generating study material: {str(e)}")

# Quiz Generator
elif option == "Quiz":
    st.subheader(f"❓ Quiz: {st.session_state.current_topic}")
    
    if st.button("Generate Quiz", type="primary"):
        with st.spinner("Generating quiz..."):
            try:
                quiz_response = quiz_agent.run(st.session_state.current_topic)
                
                if "error" in quiz_response:
                    st.error(f"Error: {quiz_response['error']}")
                    st.write("Raw response:", quiz_response['raw_response'])
                else:
                    for i, q in enumerate(quiz_response['questions'], 1):
                        st.markdown(f"**Question {i}:** {q['question']}")
                        answer = st.radio(
                            f"Select your answer for question {i}:",
                            q['options'],
                            key=f"q{i}"
                        )
                        
                        if st.button(f"Check Answer {i}", key=f"check{i}"):
                            if answer == q['correct_answer']:
                                st.success("Correct! " + q['explanation'])
                            else:
                                st.error(f"Incorrect. The correct answer is {q['correct_answer']}. " + q['explanation'])
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error generating quiz: {str(e)}")

# Flashcard Mode
elif option == "Flashcards":
    st.subheader(f"🗂 Flashcards: {st.session_state.current_topic}")
    
    if st.button("Generate Flashcards", type="primary"):
        with st.spinner("Generating flashcards..."):
            try:
                flashcards_response = flashcard_agent.run(st.session_state.current_topic)
                
                if "error" in flashcards_response:
                    st.error(f"Error: {flashcards_response['error']}")
                    st.write("Raw response:", flashcards_response['raw_response'])
                else:
                    for i, card in enumerate(flashcards_response['flashcards'], 1):
                        with st.expander(f"Flashcard {i}: {card['front']}"):
                            st.markdown(card['back'])
            except Exception as e:
                st.error(f"Error generating flashcards: {str(e)}")