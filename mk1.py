import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

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
gemini_llm = genai.GenerativeModel(model_name='gemini-1.5-pro-latest', generation_config=generation_config)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Chat interface
user_question = st.text_area("Ask a question:", key="user_question", value=st.session_state.get("user_question", ""))

ask_button = st.button("Ask", key="ask_button")

if ask_button:
    # RAG pipeline implementation
    index = pinecone.Index(pinecone_index_name)
    # Fetch relevant chunks
    xq = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query",
    )
    results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in results.matches]

    prompt_with_context = f"""{system_prompt}
    Context:
    {chr(10).join(contexts)}
    Question: {user_question}"""

    try:
        response = gemini_llm.generate_content(prompt_with_context)
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.write(f"An error occurred: {e}")
