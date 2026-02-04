import sys
from io import StringIO
from datetime import datetime
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from rag import DB_PATH
import os

RAG_ENGINE= None
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.path.join(current_dir, "knowledge_base")

def run_python_code(code: str) -> str:
    redirected_output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = redirected_output

    try:
        exec(code)
        output = redirected_output.getvalue()
        
        if not output.strip():
            output = "Code executed successfully with no output. Check your format and checkcode for return values."
    
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

    return output

def web_search(query: str):
    try:
        load_dotenv()
        ap= os.getenv("tavily_api")
        client=TavilyClient(ap)
        response=client.search(
            query=query,
            include_answer="advanced",
            search_depth="advanced",
            )
        return response
    
    except Exception as e:
        return f"Error doing web_search : {e}"
    
def get_time_date() -> str:
    return datetime.now()

def consult_archive(query: str) -> str:
    global RAG_ENGINE
    global DB_PATH
    
    if RAG_ENGINE is None:
        if os.path.exists(DB_PATH):
            print("   [Archive] 🔌 Loading Vector Database into RAM...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            RAG_ENGINE = FAISS.load_local(
                DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True 
            )
        else:
            return "Error: Knowledge base not found. Please run 'rag.py' first."
    
    print(f"   [Archive] 🧠 Searching for: '{query}'")
    results = RAG_ENGINE.similarity_search(query, k=3)
    
    # Pro-tip: Join with a separator so the LLM knows where one chunk ends and another begins
    return "\n---\n".join([doc.page_content for doc in results])

# res= consult_archive("I need the exact burst pressure specifications for the explosion vent. What is the specified burst pressure, and what is the maximum tolerance listed in the Declaration of Conformity?")
# print(res)