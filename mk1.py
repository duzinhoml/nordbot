import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Prompts
grounding_prompt = """You are a search engine that identifies the key entities and concepts in the user query related to Nord Electro 6, Nord Piano 5, Nord Stage 3, and Nord Stage 4 keyboards, as well as tone and patch creation.
User Query: {user_question}
Entities and Concepts related to Nord keyboards, tones, and patches:"""
grounding_temperature = 0.7

rag_prompt = """You are a retrieval-augmented generation system that retrieves relevant information from a Pinecone index about Nord Electro 6, Nord Piano 5, Nord Stage 3, and Nord Stage 4 keyboards, as well as tone and patch creation, including ADSR, creating pads, and other sound design topics.
User Query: {user_question}
Relevant Information about Nord keyboards and sound design:"""
rag_temperature = 0.0

synthesis_prompt = """You are a response synthesizer that combines the results from a grounding search and a RAG search to generate a final response related to Nord Electro 6, Nord Piano 5, Nord Stage 3, Nord Stage 4 keyboards, tone and patch creation, ADSR, and other sound design topics.
Grounding Search Results: {grounding_results}
RAG Search Results: {rag_results}
Final Response about Nord keyboard models or sound design:"""
synthesis_temperature = 0.4

# Streamlit UI elements
st.title("NordBot")

# Reset chat functionality
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.user_question = ""

# Pinecone configuration
pinecone_index_name = "nordbot"

# API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
serpapi_key = os.getenv("SERPAPI_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Index Settings
pinecone_dimension = 768
pinecone_metric = "cosine"
pinecone_cloud = "aws"
pinecone_region = "us-east-1"

# System Prompt
system_prompt = """You are an expert AI assistant with comprehensive and in-depth knowledge of the Nord Electro 6, Nord Piano 5, Nord Stage 3, and Nord Stage 4.

Your responses must be based exclusively on the embedded and stored documents. If a question falls outside the scope of the stored information, politely inform the user that you can only provide answers related to the available data.

Always provide accurate and precise answers, ensuring technical correctness. If the user input resembles a request for an opinion, you must provide one, but your opinion should be solely based on the information found in the stored documents.

When asked to compare or recommend a Nord keyboard for a specific use case (e.g., live performance, studio recording, versatility), provide a well-reasoned opinion based on the available documentation. Do not default to asking for clarification unless absolutely necessary.

Avoid prefacing responses with phrases such as 'According to the text,' 'Based on the information,' 'Based on the provided text,' or 'From the document.' Instead, state the answer directly and confidently.

You should be clear, concise, and professional in your responses while maintaining a knowledgeable and insightful tone. If necessary, break down complex explanations into easily digestible parts without oversimplifying important details.

Your goal is to be the ultimate source of knowledge for Nord keyboards within the constraints of the stored documents, providing **definitive recommendations and insights when applicable**.
"""

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Initialize Gemini
genai.configure(api_key=gemini_api_key)
generation_config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1096, temperature=0.0, top_p=0.7)
gemini_llm = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=generation_config)

# Initialize chat history


# Display chat messages from history on app rerun
        
# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()
input_container = st.container()
st.markdown("""
    <style>
        div.stTextInput>div>div>input {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([11, 1])

with col1:
    user_question = st.text_input("Ask a question:", key="user_question", value=st.session_state.get("user_question", ""))

with col2:
    st.markdown("""
        <style>
            div.stButton>button {

                margin-top: 12px;
            }
        </style>
    """, unsafe_allow_html=True)
    ask_button = st.button("Ask", key="ask_button")

if user_question:
    with st.spinner("Loading..."):
        # Grounding Search
        grounding_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=grounding_temperature))
        grounding_prompt_with_question = grounding_prompt.format(user_question=user_question)
        grounding_response = grounding_model.generate_content(grounding_prompt_with_question)
        grounding_results = grounding_response.text

        # RAG Search
        rag_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=rag_temperature))
        index = pinecone.Index(pinecone_index_name)
        xq = genai.embed_content(
            model="models/embedding-001",
            content=user_question if user_question else "test",
            task_type="retrieval_query",
        )
        results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
        contexts = [match.metadata['text'] for match in results.matches]
        rag_prompt_with_context = rag_prompt.format(user_question=user_question) + "\nContext:\n" + chr(10).join(contexts)
        rag_response = rag_model.generate_content(rag_prompt_with_context)
        rag_results = rag_response.text

        # Response Synthesis
        synthesis_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=synthesis_temperature))
        synthesis_prompt_with_results = synthesis_prompt.format(grounding_results=grounding_results, rag_results=rag_results)

        try:
            response = synthesis_model.generate_content(synthesis_prompt_with_results)
        except Exception as e:
            st.write(f"An error occurred: {e}")

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        try:
            response = synthesis_model.generate_content(synthesis_prompt_with_results)
            with st.chat_message("user"):
                st.write(user_question)
                st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                st.write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.write(f"An error occurred: {e}")

with input_container:
    pass

