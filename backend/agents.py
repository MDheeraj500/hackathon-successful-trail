# from langchain import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from tools import wikipedia_tool, quiz_tool, flashcard_tool, summarizer_tool

# # Initialize the OpenAI LLM model (you can adjust this for Groq or other LLMs if needed)
# llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# # Chatbot Agent for Q&A using LangChain
# qa_prompt_template = PromptTemplate(input_variables=["question"], template="Answer the following question: {question}")
# chatbot_agent = LLMChain(llm=llm, prompt=qa_prompt_template)

# # Study Material Generator Agent using LangChain
# summary_prompt_template = PromptTemplate(input_variables=["topic"], template="Summarize the topic: {topic}")
# summary_agent = LLMChain(llm=llm, prompt=summary_prompt_template)

# # Quiz Generator Agent using LangChain
# quiz_prompt_template = PromptTemplate(input_variables=["topic"], template="Generate a 5-question multiple-choice quiz on: {topic}")
# quiz_agent = LLMChain(llm=llm, prompt=quiz_prompt_template)

# # Flashcard Generator Agent using LangChain
# flashcard_prompt_template = PromptTemplate(input_variables=["topic"], template="Generate 5 flashcards on the topic: {topic}")
# flashcard_agent = LLMChain(llm=llm, prompt=flashcard_prompt_template)


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from typing import List, Dict, Optional
import json
import os
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

class GroqAgent:
    def __init__(self, groq_api_key: str, model: str = "mixtral-8x7b-32768"):
        """Initialize Groq LLM with the provided API key."""
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.7,
            max_tokens=2048
        )
        self.current_topic = None  # Add topic tracking
        
    def set_topic(self, topic: str):
        """Set the current topic for context."""
        self.current_topic = topic
        
    def _get_response(self, messages: List[dict]) -> str:
        """Helper method to get response from Groq LLM."""
        try:
            # Add topic context if available
            if self.current_topic:
                context_message = SystemMessage(content=f"The current topic is: {self.current_topic}")
                messages.insert(0, context_message)
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error getting response from Groq: {str(e)}"

    def _format_json_response(self, response: str) -> Dict:
        """Helper method to ensure response is valid JSON."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                return {
                    "error": "Could not parse response as JSON",
                    "raw_response": response
                }

    def ask_question(self, question: str, context: Optional[str] = None) -> str:
        messages = [
            SystemMessage(content="You are a helpful AI assistant that provides clear, accurate answers.")
        ]
        
        if context:
            messages.append(SystemMessage(content=f"Context: {context}"))
            
        messages.append(HumanMessage(content=question))
        return self._get_response(messages)

    def generate_summary(self, topic: Optional[str] = None) -> str:
        """Generate summary using either provided topic or current topic."""
        topic_to_use = topic or self.current_topic
        if not topic_to_use:
            return "No topic provided for summary generation."
            
        messages = [
            SystemMessage(content="""Create a comprehensive summary with the following structure:
            1. Overview
            2. Key Concepts
            3. Important Details
            4. Real-world Applications
            5. Further Learning Suggestions"""),
            HumanMessage(content=f"Please provide a detailed summary of: {topic_to_use}")
        ]
        return self._get_response(messages)

    def generate_quiz(self, topic: Optional[str] = None) -> Dict:
        """Generate quiz using either provided topic or current topic."""
        topic_to_use = topic or self.current_topic
        if not topic_to_use:
            return {"error": "No topic provided for quiz generation."}
            
        messages = [
            SystemMessage(content="""Generate 5 multiple-choice questions. Format as JSON:
            {
                "questions": [
                    {
                        "question": "Question text here",
                        "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
                        "correct_answer": "A) First option",
                        "explanation": "Explanation here"
                    }
                ]
            }"""),
            HumanMessage(content=f"Create a quiz about: {topic_to_use}")
        ]
        response = self._get_response(messages)
        return self._format_json_response(response)

    def generate_flashcards(self, topic: Optional[str] = None) -> Dict:
        """Generate flashcards using either provided topic or current topic."""
        topic_to_use = topic or self.current_topic
        if not topic_to_use:
            return {"error": "No topic provided for flashcard generation."}
            
        messages = [
            SystemMessage(content="""Generate 5 flashcards. Format as JSON:
            {
                "flashcards": [
                    {
                        "front": "Front of card text here",
                        "back": "Back of card text here"
                    }
                ]
            }"""),
            HumanMessage(content=f"Create flashcards about: {topic_to_use}")
        ]
        response = self._get_response(messages)
        return self._format_json_response(response)

class ChatbotAgent(GroqAgent):
    def __init__(self, groq_api_key: str):
        super().__init__(groq_api_key)
        
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                show_progress=False
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.doc_chain_dict = None  # Store document chain for reuse

    def init_doc_chain(self, text: str):
        """Initialize document chain for document-based conversations."""
        try:
            # Only create new chain if text has changed
            if not self.doc_chain_dict or text != getattr(self, '_last_text', None):
                chunks = self.text_splitter.split_text(text)
                vectorstore = FAISS.from_texts(chunks, self.embeddings)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful AI learning assistant. Use the following context and your knowledge about 
                    {topic} to answer the question. If you don't know the answer, just say you don't know.
                    
                    Context: {context}"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                doc_chain = create_stuff_documents_chain(
                    llm=self.llm,
                    prompt=prompt,
                    document_variable_name="context",
                )
                
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                
                chain = create_retrieval_chain(
                    retriever,
                    doc_chain,
                )
                
                self.doc_chain_dict = {
                    'chain': chain,
                    'retriever': retriever,
                    'vectorstore': vectorstore
                }
                self._last_text = text
                
            return self.doc_chain_dict
            
        except Exception as e:
            logging.error(f"Error initializing document chain: {str(e)}")
            raise

    def chat_with_docs(self, question: str, doc_chain_dict: dict, chat_history: List[Dict]) -> str:
        """Chat with document context."""
        try:
            formatted_history = []
            for i in range(0, len(chat_history)-1, 2):
                if i+1 < len(chat_history):
                    formatted_history.append(
                        (chat_history[i]["content"], chat_history[i+1]["content"])
                    )

            messages_history = []
            for human_msg, ai_msg in formatted_history:
                messages_history.append(HumanMessage(content=human_msg))
                messages_history.append(SystemMessage(content=ai_msg))

            # Include topic in the chain invocation
            response = doc_chain_dict['chain'].invoke({
                "input": question,
                "chat_history": messages_history,
                "topic": self.current_topic
            })
            
            answer = response["answer"]
            docs = doc_chain_dict['retriever'].get_relevant_documents(question)
            
            if docs:
                answer += "\n\n---\nSources:"
                used_sources = set()
                for doc in docs:
                    source_text = doc.page_content[:100] + "..."
                    if source_text not in used_sources:
                        answer += f"\n• {source_text}"
                        used_sources.add(source_text)
                
            return answer
        except Exception as e:
            logging.error(f"Error processing question with documents: {str(e)}")
            return f"Error processing question with documents: {str(e)}"

    def chat(self, question: str, chat_history: List[Dict]) -> str:
        """Regular chat without document context."""
        formatted_history = []
        for msg in chat_history[:-1]:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            else:
                formatted_history.append(SystemMessage(content=msg["content"]))

        messages = [
            SystemMessage(content=f"You are a helpful AI learning assistant focusing on {self.current_topic}. Use the chat history for context when relevant."),
            *formatted_history,
            HumanMessage(content=question)
        ]
        
        return self._get_response(messages)

class SummaryAgent(GroqAgent):
    def run(self, topic: Optional[str] = None) -> str:
        return self.generate_summary(topic)

class QuizAgent(GroqAgent):
    def run(self, topic: Optional[str] = None) -> Dict:
        return self.generate_quiz(topic)

class FlashcardAgent(GroqAgent):
    def run(self, topic: Optional[str] = None) -> Dict:
        return self.generate_flashcards(topic)