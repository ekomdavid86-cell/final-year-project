import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Setup Page
st.set_page_config(page_title="College Enquiry Bot", page_icon="🎓")
st.title("🎓 College AI Assistant")

# 2. Load Knowledge Base (Using FREE local embeddings)
@st.cache_resource
def load_knowledge_base():
    try:
        loader = TextLoader("college_info.txt", encoding="utf-8")
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # ✅ Use FREE local embeddings (no API calls!)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        return None

# 3. Initialize AI
vectorstore = load_knowledge_base()

if vectorstore:
    try:
        # ✅ Load API key from Streamlit secrets (secure method)
        google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error initializing AI: {e}")
        llm = None
        retriever = None
else:
    llm = None
    retriever = None

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_prompt := st.chat_input("Ask about fees, courses, or admission..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

# Generate Response
    if llm and retriever:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get relevant documents
                    docs = retriever.invoke(user_prompt)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Create prompt
                    full_prompt = f"""You are a helpful college admission assistant. Answer the question based on the context provided.
If you don't know the answer, say "I don't have that information" and suggest contacting the admin.

Context: {context}

Question: {user_prompt}

Answer:"""
                    
                    # ✅ Updated LLM call
                    response = llm.invoke(full_prompt).content
                    
                    # Check if AI is unsure
                    if any(phrase in response.lower() for phrase in ["i don't know", "i don't have", "contact admin"]):
                        response += "\n\n📞 **Need more help?** [Chat with Admin on WhatsApp](https://wa.me/2348000000000)"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.error("❌ Knowledge base not loaded. Please check your setup.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("AI-powered college enquiry system")
    st.write("Ask about:")
    st.write("- Admission requirements")
    st.write("- School fees")
    st.write("- Available courses")
    st.write("- Campus facilities")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()