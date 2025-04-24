# import streamlit as st
# import pandas as pd
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # UI Styling
# st.markdown("""
# <style>
#     .stApp {
#         background: #f8f5e6;
#         background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
#         background-size: 20px 20px;
#     }
#     .chat-font {
#         font-family: 'Times New Roman', serif;
#         color: #2c5f2d;
#     }
#     .user-msg {
#         background: #ffffff !important;
#         border-radius: 15px !important;
#         border: 2px solid #2c5f2d !important;
#     }
#     .bot-msg {
#         background: #fff9e6 !important;
#         border-radius: 15px !important;
#         border: 2px solid #ffd700 !important;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Configure Gemini API
# genai.configure(api_key="AIzaSyALSp9jVTV04ziisP5IdFu5-VpPsx39NfU")
# gemini = genai.GenerativeModel('gemini-1.5-flash')

# # Load the embedder
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# # Load the dataset and FAISS index
# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_csv('my_data_chatbot.csv')  # Ensure the CSV has 'question' and 'answer' columns
#         if 'question' not in df.columns or 'answer' not in df.columns:
#             st.error("The CSV must contain 'question' and 'answer' columns.")
#             st.stop()
#         df['context'] = df.apply(lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", axis=1)
#         embeddings = embedder.encode(df['context'].tolist())
#         index = faiss.IndexFlatL2(embeddings.shape[1])
#         index.add(np.array(embeddings).astype('float32'))
#         return df, index
#     except Exception as e:
#         st.error(f"Failed to load data: {e}")
#         st.stop()

# df, faiss_index = load_data()

# # Title and instructions
# st.markdown('<h1 class="chat-font">🤖 Satvika Chatbot</h1>', unsafe_allow_html=True)
# st.markdown("<h3 class='chat-font'>Ask me anything, and I’ll respond as Satvika!</h3>", unsafe_allow_html=True)
# st.markdown("---")

# # Helper functions
# def find_closest_question(query, faiss_index, df):
#     query_embedding = embedder.encode([query])
#     _, I = faiss_index.search(query_embedding.astype('float32'), k=1)
#     if I.size > 0:
#         return df.iloc[I[0][0]]['answer']
#     return None

# def generate_refined_answer(query, retrieved_answer, chat_history):
#     chat_context = "\n".join([f"User: {msg['content']}" for msg in chat_history if msg["role"] == "user"][-5:])
#     prompt = f"""
#     You are Satvika, an AIML student. Reply in a friendly, engaging tone.
#     Chat History:\n{chat_context}
#     Current Question: {query}
#     Retrieved Answer: {retrieved_answer}
    
#     - Be clear, friendly, and engaging.
#     - Use proper grammar.
#     - Make sure the answer is at least 3 lines.
#     """
#     response = gemini.generate_content(prompt)
#     return response.text.strip() if response else retrieved_answer

# # Maintain chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"], avatar="🙋" if message["role"] == "user" else "🤖"):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask me anything..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     with st.spinner("Thinking..."):
#         try:
#             retrieved_answer = find_closest_question(prompt, faiss_index, df)
#             if retrieved_answer:
#                 response = generate_refined_answer(prompt, retrieved_answer, st.session_state.messages)
#             else:
#                 response = "I'm still learning and don't have an answer for that right now!"
#         except Exception as e:
#             response = f"Oops! Something went wrong: {e}"
    
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.rerun()

import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# UI Styling
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .chat-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY_HERE")  # Update with your real API key
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load the embedder and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load the dataset and FAISS index
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('my_data_chatbot.csv')  # Ensure the CSV has 'question' and 'answer' columns
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", axis=1)
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

df, faiss_index = load_data()

# Title and instructions
st.markdown('<h1 class="chat-font">🤖 Satvika Chatbot</h1>', unsafe_allow_html=True)
st.markdown("<h3 class='chat-font'>Ask me anything, and I’ll respond as Satvika!</h3>", unsafe_allow_html=True)
st.markdown("---")

# Helper functions
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']
    return None

def generate_refined_answer(query, retrieved_answer, chat_history):
    chat_context = "\n".join([f"User: {msg['content']}" for msg in chat_history if msg["role"] == "user"][-5:])
    prompt = f"""
    You are Satvika, an AIML student. Reply in a friendly, engaging tone.
    Chat History:\n{chat_context}
    Current Question: {query}
    Retrieved Answer: {retrieved_answer}
    
    - Be clear, friendly, and engaging.
    - Use proper grammar.
    - Make sure the answer is at least 3 lines.
    """
    response = gemini.generate_content(prompt)
    return response.text.strip() if response else retrieved_answer

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                response = generate_refined_answer(prompt, retrieved_answer, st.session_state.messages)
            else:
                response = "I'm still learning and don't have an answer for that right now!"
        except Exception as e:
            response = f"Oops! Something went wrong: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
