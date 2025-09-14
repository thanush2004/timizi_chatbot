import os
import textwrap
import asyncio
import io
import sys
import requests
import json
import re
import threading
from PyPDF2 import PdfReader
import google.generativeai as genai
from supabase import create_client, Client
from supabase.client import AsyncClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from enum import Enum, auto
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_talisman import Talisman

# --- Enums and State Classes ---
class ConversationState(Enum):
    """
    Manages the conversational state of the chatbot.
    """
    INITIAL = auto()
    PROMPTING_SKIN_TYPE = auto()
    PROMPTING_SKIN_PROBLEM = auto()
    PROMPTING_SKIN_LEVEL = auto()
    PROMPTING_HAIR_TYPE = auto()
    PROMPTING_HAIR_LEVEL = auto()
    COMPLETE = auto()

class Prompter:
    """
    A base class for handling a sequence of questions and user responses.
    """
    def __init__(self, supabase_client):
        self.supabase_client = supabase_client
        self.state = ConversationState.INITIAL
        self.answers = {}

    async def get_next_question(self, user_response: str) -> dict:
        """
        Processes a user response and returns the next question or a final response.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError

    async def get_product_tags(self) -> list:
        """
        Generates product tags from the collected answers.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError

class SkinCarePrompter(Prompter):
    """
    Handles the multi-turn conversation for skin care product recommendations.
    """
    def __init__(self, supabase_client):
        super().__init__(supabase_client)
        self.product_keywords = {
            "oily": "oily", "dry": "dry", "normal": "normal", "sensitive": "sensitive", "combination": "combination",
            "acne": "acne-pimples", "pimples": "acne-pimples", "pigmentation": "pigmentation-dark-spots",
            "dark circle": "dark-circle", "tanning": "tanning-uneven-skinton", "glow": "glow", "clear": "clear",
            "mild": "mild", "moderate": "moderate", "severe": "severe", "healing": "healing", "zits": "acne-pimples"
        }

    async def get_next_question(self, user_response: str):
        """
        Processes a user response and returns the next question or a final response.
        """
        if self.state == ConversationState.INITIAL:
            self.state = ConversationState.PROMPTING_SKIN_TYPE
            return {"type": "text", "response": "What is your skin type? (e.g., normal, combination, dry, sensitive, or oily)"}

        elif self.state == ConversationState.PROMPTING_SKIN_TYPE:
            user_response_lower = user_response.lower()
            matching_key = next((key for key in self.product_keywords if key in user_response_lower), None)
            if matching_key:
                self.answers['skin_type'] = self.product_keywords[matching_key]
                self.state = ConversationState.PROMPTING_SKIN_PROBLEM
                return {"type": "text", "response": "What is your main skin problem? (e.g., acne, pigmentation, dark circles, tanning)"}
            else:
                return {"type": "text", "response": "I didn't understand that. Please choose from: normal, combination, dry, sensitive, or oily."}

        elif self.state == ConversationState.PROMPTING_SKIN_PROBLEM:
            user_response_lower = user_response.lower()
            matching_key = next((key for key in self.product_keywords if key in user_response_lower), None)
            if matching_key:
                self.answers['skin_problem'] = self.product_keywords[matching_key]
                self.state = ConversationState.PROMPTING_SKIN_LEVEL
                return {"type": "text", "response": "What is the problem level? (e.g., mild, moderate, severe, or healing)"}
            else:
                return {"type": "text", "response": "I didn't understand that. Please specify a problem like acne, pigmentation, or dark circles."}

        elif self.state == ConversationState.PROMPTING_SKIN_LEVEL:
            user_response_lower = user_response.lower()
            matching_key = next((key for key in self.product_keywords if key in user_response_lower), None)
            if matching_key:
                self.answers['problem_level'] = self.product_keywords[matching_key]
                self.state = ConversationState.COMPLETE
                product_tags = await self.get_product_tags()
                products = await search_products_by_tags(self.supabase_client, product_tags)
                return products
            else:
                return {"type": "text", "response": "I didn't understand that. Please specify the level: mild, moderate, severe, or healing."}
        
        return {"type": "text", "response": "Sorry, something went wrong. Let's start over. What are you looking for?"}

    async def get_product_tags(self) -> list:
        """
        Generates product tags from the collected answers.
        """
        tags = []
        if 'skin_type' in self.answers:
            tags.append(self.answers['skin_type'])
        if 'skin_problem' in self.answers:
            tags.append(self.answers['skin_problem'])
        if 'problem_level' in self.answers:
            tags.append(self.answers['problem_level'])
        return tags

class HairCarePrompter(Prompter):
    """
    Handles the multi-turn conversation for hair care product recommendations.
    """
    def __init__(self, supabase_client):
        super().__init__(supabase_client)
        self.product_keywords = {
            "hair fall": "hairfall", "hairfall": "hairfall", "dandruff": "dandruff",
            "oily scalp": "oily", "slow hair growth": "slow-hair-growth", "dry": "dry-frizz",
            "frizzy": "dry-frizz", "mild": "mild", "moderate": "moderate", "severe": "severe",
            "healing": "healing"
        }

    async def get_next_question(self, user_response: str):
        """
        Processes a user response and returns the next question or a final response.
        """
        if self.state == ConversationState.INITIAL:
            self.state = ConversationState.PROMPTING_HAIR_TYPE
            return {"type": "text", "response": "What is your main hair concern? (e.g., hair fall, dandruff, oily scalp, slow hair growth, dry/frizzy)"}

        elif self.state == ConversationState.PROMPTING_HAIR_TYPE:
            user_response_lower = user_response.lower()
            matching_key = next((key for key in self.product_keywords if key in user_response_lower), None)
            if matching_key:
                self.answers['hair_type'] = self.product_keywords[matching_key]
                self.state = ConversationState.PROMPTING_HAIR_LEVEL
                return {"type": "text", "response": "What is the problem level? (e.g., mild, moderate, severe, or healing)"}
            else:
                return {"type": "text", "response": "I didn't understand that. Please specify a hair concern like hair fall, dandruff, or dry/frizzy."}

        elif self.state == ConversationState.PROMPTING_HAIR_LEVEL:
            user_response_lower = user_response.lower()
            matching_key = next((key for key in self.product_keywords if key in user_response_lower), None)
            if matching_key:
                self.answers['problem_level'] = self.product_keywords[matching_key]
                self.state = ConversationState.COMPLETE
                product_tags = await self.get_product_tags()
                products = await search_products_by_tags(self.supabase_client, product_tags)
                return products
            else:
                return {"type": "text", "response": "I didn't understand that. Please specify the level: mild, moderate, severe, or healing."}
        
        return {"type": "text", "response": "Sorry, something went wrong. Let's start over. What are you looking for?"}

    async def get_product_tags(self) -> list:
        """
        Generates product tags from the collected answers.
        """
        tags = []
        if 'hair_type' in self.answers:
            tags.append(self.answers['hair_type'])
        if 'problem_level' in self.answers:
            tags.append(self.answers['problem_level'])
        return tags

# --- Global State for Conversation ---
conversation_prompter = None

# --- Helper Functions ---
def get_or_create_supabase_client():
    """
    Initializes and returns an async Supabase client for version 2.x.
    Requires SUPABASE_URL and SUPABASE_ANON_KEY to be set in .env
    """
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key:
            raise KeyError("SUPABASE_URL or SUPABASE_ANON_KEY not set.")

        return AsyncClient(supabase_url, supabase_key)
    except KeyError as e:
        print(f"Error: {e}")
        print("Please ensure your .env file has SUPABASE_URL and SUPABASE_ANON_KEY.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while connecting to Supabase: {e}")
        sys.exit(1)

def get_embedding(text, model="models/embedding-001", task_type="retrieval_document"):
    """
    Generates an embedding for a given text using the Gemini model.
    """
    try:
        response = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def contains_product_keywords(query: str) -> bool:
    """
    Checks if a user query contains keywords related to products.
    """
    product_keywords = ["product", "serum", "cream", "shampoo", "oil", "treatment", "moisturizer", "sunscreen", "toner", "cleanser", "makeup", "for skin", "for hair", "concern"]
    return any(keyword in query.lower() for keyword in product_keywords)

def contains_specific_product_type(query: str) -> str | None:
    """
    Checks if a user query contains a specific product type and returns it.
    """
    product_types = ["facewash", "cleanser", "toner", "moisturizer", "serum", "mask", "exfoliator", "sunscreen", "shampoo"]
    for product_type in product_types:
        if product_type in query.lower():
            return product_type
    return None

async def ingest_document(supabase, document_record: dict, chunk_table_name: str):
    """
    Processes a single document record from the chatbot_documents table,
    creates chunks and embeddings, and stores them in the document_chunks table.
    """
    file_url = document_record['file_url']
    file_name = document_record['file_name']
    file_visibility = document_record['visibility']
    print(f"Processing new file: {file_name}")

    try:
        # Download the file from the URL
        print("Downloading file...")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        file_content = response.content

        # Use PyPDF2 to read the content from the bytes directly
        reader = PdfReader(io.BytesIO(file_content))
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)

        # Create embeddings for each chunk
        data_to_upsert = []
        for chunk in chunks:
            embedding = get_embedding(chunk, model="models/embedding-001", task_type="retrieval_document")
            if embedding:
                data_to_upsert.append({
                    "file_name": file_name,
                    "content": chunk,
                    "visibility": file_visibility,
                    "embedding": embedding
                })

        # Upsert the documents and embeddings into the new table
        if data_to_upsert:
            await supabase.table(chunk_table_name).upsert(data_to_upsert).execute()
            print(f"Upserted {len(data_to_upsert)} chunks from {file_name} to the '{chunk_table_name}' table.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file {file_name}: {e}")
    except Exception as e:
        print(f"An error occurred during document processing: {e}")

async def listen_for_new_documents(supabase, doc_table_name: str, chunk_table_name: str):
    """
    Sets up a real-time listener to ingest new documents as they are uploaded.
    This uses the correct client.stream() method for Supabase v2.x.
    """
    print(f"Listening for new documents in '{doc_table_name}'...")
    
    # Initial fetch to process any existing documents not yet ingested
    print("Performing initial ingestion of all existing documents...")
    try:
        files_data = await supabase.table(doc_table_name).select('file_url, file_name, visibility').execute()
        for record in files_data.data:
            await ingest_document(supabase, record, chunk_table_name)
    except Exception as e:
        print(f"Initial ingestion failed: {e}")
        
    # Start the real-time stream listener
    try:
        async for event in supabase.from_(doc_table_name).stream():
            if event.get('type') == 'INSERT':
                new_record = event.get('new')
                print(f"New file detected: {new_record.get('file_name')}")
                await ingest_document(supabase, new_record, chunk_table_name)
    except Exception as e:
        print(f"An error occurred in the real-time listener: {e}")

async def search_products(supabase, query: str):
    """
    Starts the conversational flow for a product search.
    """
    global conversation_prompter
    if "hair" in query.lower():
        conversation_prompter = HairCarePrompter(supabase)
    else:
        conversation_prompter = SkinCarePrompter(supabase)
    
    return await conversation_prompter.get_next_question(query)

async def search_products_by_tags(supabase, tags: list):
    """
    Searches the products table based on a list of tags.
    """
    try:
        response = await supabase.table("product").select("*").overlaps('categories', tags).limit(10).execute()
        
        if response.data:
            return {"type": "product_list", "products": response.data}
        else:
            return {"type": "text", "response": "Sorry, I couldn't find any products matching your specific needs. You might want to try a different combination of preferences."}

    except Exception as e:
        print(f"An error occurred during product search by tags: {e}")
        return {"type": "text", "response": "I'm having trouble searching for products right now. Please try again later."}
    
async def search_by_subtype(supabase, subtype: str):
    """
    Searches the products table based on a specific product subtype.
    """
    try:
        # Use 'overlaps' to find products where the 'categories' array contains the subtype
        response = await supabase.table("product").select("*").overlaps('categories', [subtype.lower()]).limit(10).execute()
        
        if response.data:
            return {"type": "product_list", "products": response.data}
        else:
            return {"type": "text", "response": f"Sorry, I couldn't find any products of type '{subtype}'."}
    except Exception as e:
        print(f"An error occurred during subtype search: {e}")
        return {"type": "text", "response": "I'm having trouble searching for products right now. Please try again later."}


async def ask_chatbot(supabase, user_query: str, chunk_table_name: str):
    """
    Answers a user query by retrieving relevant context from Supabase and generating a response.
    Prioritizes product queries before attempting document-based retrieval.
    """
    global conversation_prompter

    # Check for a specific product type first
    product_subtype = contains_specific_product_type(user_query)
    if product_subtype:
        conversation_prompter = None
        return await search_by_subtype(supabase, product_subtype)
    
    # Continue a product recommendation conversation if in progress
    if conversation_prompter and conversation_prompter.state != ConversationState.COMPLETE:
        return await conversation_prompter.get_next_question(user_query)
    
    # Start a new product recommendation conversation if keywords are present
    if contains_product_keywords(user_query):
        return await search_products(supabase, user_query)
    
    # If no product keywords, fall back to document-based RAG
    # 1. Embed the user's query for document search
    embedding_model = "models/embedding-001"
    query_embedding = get_embedding(user_query, model=embedding_model, task_type="retrieval_query")
    if not query_embedding:
        return {"type": "text", "response": "I'm sorry, I couldn't process your query."}

    # 2. Query the Supabase database for the most relevant chunk using the `match_documents` RPC function
    try:
        response = await supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': 1
            }
        ).execute()

        if response.data and len(response.data) > 0:
            retrieved_context = response.data[0]['content']
        else:
            return {"type": "text", "response": "Sorry, I couldn't find any relevant information in the knowledge base."}

    except Exception as e:
        print(f"An error occurred during Supabase RPC call: {e}")
        return {"type": "text", "response": "I'm having trouble accessing my knowledge base. Please try again later."}
    
    print(f"\nRetrieved Context:\n{textwrap.fill(retrieved_context, 70)}")

    # 3. Construct the prompt with the retrieved context
    prompt = f"""
    You are a helpful assistant. Use only the following context to answer the question. If the answer is not in the context, say that you don't have the information.

    Context:
    {retrieved_context}

    User Question:
    {user_query}
    """
    
    # 4. Call the Gemini API's generative model
    generation_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    try:
        response = generation_model.generate_content(prompt)
        return {"type": "text", "response": response.text}
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        return {"type": "text", "response": "I'm currently unable to generate a response. Please try again later."}

# --- Flask App Configuration and Routes ---
app = Flask(__name__)
CORS(app)
Talisman(app) # Adds security headers

@app.route('/ask_chatbot', methods=['POST'])
async def handle_chat_request():
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({'type': 'text', 'response': 'No query provided.'}), 400
    
    response_data = await ask_chatbot(supabase_client, user_query, SUPABASE_CHUNK_TABLE)
    
    return jsonify(response_data)

# --- Server Startup Logic ---
def start_async_listener(loop):
    """Helper function to run the async listener in a separate thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(listen_for_new_documents(supabase_client, SUPABASE_DOCUMENT_TABLE, SUPABASE_CHUNK_TABLE))
    except (KeyboardInterrupt, SystemExit):
        print("Listener thread shutting down.")
        
if __name__ == '__main__':
    load_dotenv()
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        print("Error: 'GEMINI_API_KEY' environment variable not set.")
        sys.exit(1)
    
    SUPABASE_CHUNK_TABLE = "document_chunks"
    SUPABASE_DOCUMENT_TABLE = "chatbot_documents"

    # Initialize the async Supabase client
    supabase_client = get_or_create_supabase_client()
    
    if supabase_client:
        # Create a new event loop for the async listener
        listener_loop = asyncio.new_event_loop()
        # Start the listener in a new thread
        listener_thread = threading.Thread(target=start_async_listener, args=(listener_loop,), daemon=True)
        listener_thread.start()
        
        # This line is for local
