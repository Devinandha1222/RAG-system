import gradio as gr
from dotenv import load_dotenv
import os
import logging
import sys
import base64
import math
import json 

# --- Suppress Warnings ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)


vector_store = None 
web_search_tool = DuckDuckGoSearchRun()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

current_interaction_state = {"query": "", "category": ""}
RL_DATA_FILE = "rl_good_examples.json"


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.4
)


def load_good_examples():
    if not os.path.exists(RL_DATA_FILE): return ""
    try:
        with open(RL_DATA_FILE, "r") as f:
            data = json.load(f)
            return "\n".join([f"- Query: '{item['query']}' -> Category: {item['category']}" for item in data[-5:]]) 
    except: return ""

def save_feedback(query, category, feedback_type):
    if feedback_type != "like": return 
    new_entry = {"query": query, "category": category}
    data = []
    if os.path.exists(RL_DATA_FILE):
        with open(RL_DATA_FILE, "r") as f:
            try: data = json.load(f)
            except: data = []
    if new_entry not in data:
        data.append(new_entry)
        with open(RL_DATA_FILE, "w") as f:
            json.dump(data, f)
        print(f"🧠 RL Update: Learned strategy for '{query}'")


def describe_image_with_gemini(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe text exactly. Describe charts/visuals."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        )
        return llm.invoke([message]).content
    except: return ""

def process_files(files):
    global vector_store
    if not files: return "⚠️ No files uploaded."
    documents = []
    status_log = []
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            lower_name = filename.lower()
            if lower_name.endswith('.pdf'): documents.extend(PyPDFLoader(file_path).load())
            elif lower_name.endswith('.docx'): documents.extend(Docx2txtLoader(file_path).load())
            elif lower_name.endswith('.txt'): documents.extend(TextLoader(file_path, encoding='utf-8').load())
            elif lower_name.endswith('.pptx'): documents.extend(UnstructuredPowerPointLoader(file_path).load())
            elif lower_name.endswith(('.jpg', '.png')):
                desc = describe_image_with_gemini(file_path)
                if desc: documents.append(Document(page_content=desc, metadata={"source": filename}))
            status_log.append(f"✅ Loaded: {filename}")
        except Exception as e: status_log.append(f"❌ Error {filename}: {str(e)}")

    if not documents: return "\n".join(status_log) + "\n\n⚠️ No valid text."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    if vector_store is None: vector_store = FAISS.from_documents(chunks, embeddings)
    else: vector_store.add_documents(chunks)
    
    return "\n".join(status_log) + f"\n\n🚀 Ready! Indexed {len(chunks)} chunks."

def safe_calculator(expression):
    try:
        clean_prompt = ChatPromptTemplate.from_template("Extract math from: '{text}'. Return ONLY numbers/operators.")
        cleaned_math = (clean_prompt | llm | StrOutputParser()).invoke({"text": expression}).strip()
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        return f"Result: {eval(cleaned_math, {'__builtins__': {}}, allowed)}"
    except: return "I couldn't solve that math problem safely."



# A. Sentiment Analyzer
sentiment_prompt = ChatPromptTemplate.from_template(
    """
    Analyze the sentiment of this user query: "{query}"
    Return a brief description of the emotional state (e.g., Frustrated, Curious).
    Format: State
    """
)
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# B. Classifier
classify_prompt = ChatPromptTemplate.from_template(
    """
    Analyze the user query: "{query}"
    PAST EXAMPLES: {learned_examples}
    
    Classify into exactly one:
    1. "Retrieval_Required" (Questions about uploaded files/specific docs).
    2. "Web_Search" (General knowledge, news, facts not in personal docs).
    3. "Calculation" (Math).
    4. "Chit_Chat" (Greetings, small talk).
    
    Return ONLY the category name.
    """
)
classifier_chain = classify_prompt | llm | StrOutputParser()

# C. Router
def router_execution(inputs):
    global vector_store
    query = inputs["query"]
    category = inputs["category"]
    
    if category == "Retrieval_Required" and vector_store is None:
        category = "Web_Search" 

    if category == "Chit_Chat": 
        return "No external context needed."
    elif category == "Web_Search": 
        try:
            results = web_search_tool.invoke(query)
            return f"WEB SEARCH RESULTS (Use these to answer and cite sources):\n{results}"
        except: return "Web search failed."
    elif category == "Calculation": 
        return safe_calculator(query)
    elif category == "Retrieval_Required":
        results = vector_store.similarity_search(query, k=3)
        return f"DOCUMENT CONTEXT:\n" + "\n".join([d.page_content for d in results])
    
    return "No category."

# D. Generator
generate_prompt = ChatPromptTemplate.from_messages([
    # 1. The System Message: Defines WHO the AI is.
    ("system", "{persona_instruction}"),
    
    # 2. The Human Message: Contains the data and query.
    ("human", """
    USER QUERY: {query}
    USER SENTIMENT: {sentiment}
    
    CONTEXT INFORMATION: 
    {context}
    
    INSTRUCTIONS:
    1. If Context contains Web Search Results, answer and list CITATIONS.
    2. If Context is Documents and you are in "ROAST" mode, critique the document content/formatting first.
    3. Be useful, but follow your persona strictly.
    """)
])
final_chain = generate_prompt | llm | StrOutputParser()

def sequential_chat_logic(message, history, mode_selection):
    global current_interaction_state
    try:
        # Match string exactly with the UI Radio Value
        if mode_selection == "Roast Master":
            persona = (
                "You are a sarcastic, witty, but highly intelligent AI with immense knowledge"
                "you can explain any complex topic in simple and understanding terms"
                "If the user provides context from a document, you can roast the document first if it is bad. "
                "Make comments like  'Did a toddler write this?', be mild"
                "or 'This table is so misaligned even I can't read it'. "
                "After the roast, answer the question correctly but with sass." 
                "crack jokes related to the content where appropriate."
            )
        else:
            persona = (
                "You are an empathetic, professional, and academic assistant. "
                "Adapt your tone to be helpful and reassuring based on the user's sentiment. "
                "Provide clear, concise, and accurate answers."
                "Students use you to learn complex topics, so break down explanations step-by-step."
                "Provide easy to understand examples where applicable."
                "While quoting from web search results, ensure to cite sources appropriately.provide links of sites,document names etc"
            )

        sentiment_info = sentiment_chain.invoke({"query": message})
        learned_examples = load_good_examples()
        category = classifier_chain.invoke({"query": message, "learned_examples": learned_examples}).strip()
        
        final_category = category
        if category == "Retrieval_Required" and vector_store is None:
            final_category = "Web_Search"
            
        context = router_execution({"query": message, "category": category})
        
        current_interaction_state["query"] = message
        current_interaction_state["category"] = final_category
        
        response = final_chain.invoke({
            "query": message, 
            "sentiment": sentiment_info,
            "context": context,
            "persona_instruction": persona
        })
        
        emoji_map = {"Retrieval_Required": "📄", "Web_Search": "🌐", "Calculation": "🧮", "Chit_Chat": "💬"}
        icon = emoji_map.get(final_category, "🤖")
        
        debug_msg = f"\n\n---\n* {sentiment_info} | Action: {icon} {final_category} | Mode: {mode_selection}*"
        
        return response + debug_msg
    except Exception as e:
        return f"System Error: {e}"

# --- 7. Feedback Handler ---
def on_feedback_vote(data: gr.LikeData):
    if data.liked:
        print(f"👍 User liked: {current_interaction_state['query']}")
        save_feedback(current_interaction_state["query"], current_interaction_state["category"], "like")

# --- 8. UI Setup ---
with gr.Blocks(theme=gr.themes.Soft(), title="Dual-Mode Agent") as demo:
    gr.Markdown(" The Agent")
    gr.Markdown("Switch modes below to change how I treat you and your files.")
    
    with gr.Row():
        with gr.Column(scale=1):
            mode_radio = gr.Radio(
                choices=["Serious Study", "Roast Master"], 
                value="Serious Study", 
                label=" Personality Mode",
                info="Choose 'Roast Master' if you want me to judge your documents."
            )
            
            file_input = gr.File(label="📂 Upload Files", file_count="multiple", type="filepath")
            process_btn = gr.Button("Process Files", variant="primary")
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            process_btn.click(process_files, file_input, status)

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=sequential_chat_logic,
                additional_inputs=[mode_radio], 
                title="Chat with the Agent",
            )
            chatbot.chatbot.like(on_feedback_vote, None, None)

if __name__ == "__main__":
    demo.launch()