# RAG-system

**Problem Statement**

Modern learners interact with heterogeneous study material which includes textual documents like PDF, DOCX, TXT, PPTX and visual contents like diagrams, notes, screenshots. Traditional information retrieval systems and conversational AI models operate in isolation and they lack the capability to integrate multimodal data sources for contextual learning assistance.
Most of the models have upload limits in case of documents and images.
Furthermore, existing RAG systems are primarily text-centric and fail to incorporate image understanding, adaptive personas, sentiment-aware responses, or dynamic task routing based on user intent.

This creates a gap in providing an intelligent, multimodal retrieval and generation framework capable of assisting students with personalized academic queries grounded in their own study materials.

The core research problem is:

How can we design a multimodal Retrieval-Augmented Generation agent that integrates document parsing, image understanding, vector-based retrieval, web search, and LLM reasoning to deliver accurate, context-aware, and sentiment-adjusted study assistance?

This project addresses this problem by developing an orchestrated AI agent equipped with:
multimodal ingestion and indexing
classification-based task routing
sentiment-driven response adaptation
persona-switching for conversations
reinforcement-learning–based user feedback improvement

The goal is to create a robust, context-aware study assistant capable of answering queries derived from both textual and visual academic resources.

 **Why agents?** 

Agents are the right solution for this problem because the project requires intelligent decision making, multimodal processing, and tool orchestration. Students ask many different types of questions, and an agent can classify intent, select the correct tool (document retrieval, image analysis, web search, calculator), and generate accurate context-aware answers. Agents provide modularity, adaptability, persona switching, sentiment-based responses, and continuous learning, making them the ideal architecture for building a flexible multimodal study assistant.

A normal LLM is insufficient for this project because it cannot read documents, analyze images, perform web search, run tools, retrieve vector-store data, classify user intent, switch personas, adapt to sentiment, or learn from feedback. An agent-based system is required to orchestrate tools, route tasks intelligently, and deliver multimodal academic assistance, which a stand-alone LLM simply cannot achieve.

**Architecture**

The project implements a Multi-Modal Study Agent, built on top of a RAG pipeline and extended using agentic orchestration. The system combines documents, images, tool-using agents, and a central orchestrator to deliver personalized study assistance.

1. The user interacts with the system through:
Chat UI
Upload interface for PDFs, text files, images and other documents
This layer sends all requests to the Orchestrator Agent.
2. Orchestrator 
Interprets user intent
Decides whether to call document RAG, image analysis, or other tools
Coordinates multiple sub-agents
Collects all partial outputs
Produces a single, blended final answer

The orchestrator uses:
LLM reasoning
Tool selection logic
Routing capabilities

3.Document Processing Pipeline (Text RAG Agent)
Handles all uploaded documents.
Components:
Text Extractor (PDF loader / OCR if needed)
Text Splitter (Recursive splitting for long content)
Embeddings Generator (HuggingFace/OpenAI embeddings)
Vector Store (e.g., FAISS)
Retriever (top-k similarity search)

4.Image Analysis Pipeline (Vision Agent)
Handles:
Diagrams
Charts
Graphs
Images of handwritten notes
Components:
Vision model (Gemini Vision, GPT-4o mini, etc.)
Image-to-text interpretation
OCR layer

5.Storage Layer
Stores:
Vector embeddings
Uploaded documents
Parsed text
Extracted images/content
Metadata such as filenames, topics, keywords

**Workflow**

1. Input Collection
The user submits a question or uploads documents/images. All inputs are sent to the Orchestrator Agent.

2. Intent Detection
The Orchestrator identifies the user’s goal and decides whether to use the document RAG pipeline, image analysis pipeline, or a specialized tool.

3. Document Processing (RAG)
If documents are uploaded, the system extracts text, chunks it, generates embeddings, stores them in a vector database, and retrieves relevant sections based on the query.

4. Image Processing
For images, a vision-language model performs OCR, interprets diagrams or handwritten notes, and converts them into structured text.

5. Tool Invocation
The Orchestrator may call tools such as summarization, question generation, explanation, or study planning depending on the user’s task.

6. Response Synthesis
The LLM combines retrieved text, processed images, and tool outputs to generate a coherent and grounded final answer.

7. Output Delivery
The final response summary, explanation, Q&A set, or study notes—is delivered to the user, with context preserved for follow-up questions.

**Tech stack**
1. Programming Language
Python :Core development language for backend logic, integrations, and agent orchestration.

2. Frameworks & Libraries
Gradio : Interactive UI for chatting, file uploads, and agent interaction.
LangChain : Agent workflow, tools, prompts, routing, and RAG pipeline.
SentenceTransformers (HuggingFace) :Embeddings for vector search.
FAISS :Fast similarity search and vector storage.
dotenv : Secure environment variable management(invoking API).


3. Language Model
Google Gemini 2.5 Flash : For reasoning, generation, image understanding, routing, OCR-like extraction, and persona responses.

4. Document & Image Processing
PyPDFLoader : PDF extraction
Docx2txtLoader :DOCX extraction
TextLoader :TXT ingestion
UnstructuredPowerPointLoader : PPTX extraction
Base64 Image Encoding : For sending images to Gemini
Vision capabilities of Gemini : For diagram + text extraction from images

5. Search & Retrieval
FAISS Vector Store : Document chunk storage and similarity retrieval
HuggingFace Embeddings Model-all-MiniLM-L6-v2 : Creating vector embeddings
DuckDuckGo Search Tool :Web search integration for external knowledge

6. Agent Logic
LangChain Tools : For search, calculations, and routing
Classification Chain : Intent categorization
Sentiment Chain : Emotional tone detection
Custom Router : For deciding Retrieval vs Web Search vs Calculation
Feedback Learning : Reinforcement-like improvement using rl_good_examples.json

7. UI/Frontend
Gradio ChatInterface & Blocks : Interactive chat, file upload area, feedback buttons, mode selection..

**Future improvements**

Enhance image understanding to support:
Mathematical equation detection & solving
Diagram-to-text and diagram-to-explanation translation
Table extraction from images/PDFs

Allow the system to learn:
User’s weak topics
Preferred explanation style
Pace of study
Then generate personalized study plans and adaptive recommendations.

Enable support for:
Indian regional languages
PDF/Image extraction in bilingual content
Useful for students studying in vernacular media.

Add:
Encrypted document storage
Access control
Local-only processing mode
Useful for sensitive academic documents.
