import os
import uuid
import time
import json
import logging
import requests
import google.generativeai as genai
import numpy as np
from mem.emb import get_embedding, cosine_similarity
from mem.pine_client import index
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
import re
import agno
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.sql import SQLTools
import psycopg2
from psycopg2 import sql
from textwrap import dedent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Get all available Gemini API keys
def get_gemini_api_keys():
    keys = []
    # Get the primary key
    primary_key = os.getenv("GEMINI_API_KEY")
    if primary_key:
        keys.append(primary_key)

    # Get additional keys that might be in the .env file
    for i in range(1, 10):  # Check for up to 10 additional keys
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)

    # Get the Google API key as a fallback
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key and google_api_key not in keys:
        keys.append(google_api_key)

    # Check for keys that might be directly in the .env file without a variable name
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                # Look for lines that contain just an API key (starts with AIza)
                if line.startswith("AIza") and len(line) > 30 and "=" not in line:
                    if line not in keys:
                        keys.append(line)
                # Also check for keys in format GOOGLE_API_KEY=AIza...
                elif "=" in line and "API_KEY" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2 and parts[1].startswith("AIza") and len(parts[1]) > 30:
                        key = parts[1].strip()
                        if key not in keys:
                            keys.append(key)
    except Exception as e:
        logger.error(f"Error reading .env file: {str(e)}")

    logger.info(f"Found {len(keys)} Gemini API keys")
    return keys

# Available Gemini models to use with key rotation
GEMINI_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

# Initialize Gemini model with key rotation
gemini_api_keys = get_gemini_api_keys()
current_key_index = 0
current_model_index = 0

def get_gemini_model(model_name=None):
    global current_key_index, current_model_index, gemini_api_keys

    # If we've tried all keys, start over
    if current_key_index >= len(gemini_api_keys):
        current_key_index = 0
        logger.warning("All Gemini API keys have been tried and failed. Starting over.")

    # Get the current key
    api_key = gemini_api_keys[current_key_index]
    logger.info(f"Using Gemini API key at index {current_key_index}")

    # Configure the genai client
    genai.configure(api_key=api_key)

    # If model name is provided, use it
    if model_name:
        return genai.GenerativeModel(model_name)

    # Otherwise use the current model from rotation
    model_to_use = GEMINI_MODELS[current_model_index]
    logger.info(f"Using Gemini model: {model_to_use}")

    # Create and return the model
    return genai.GenerativeModel(model_to_use)

# Function to rotate to the next key when one fails
def rotate_gemini_key():
    global current_key_index

    # Move to the next key
    current_key_index += 1
    logger.info(f"Rotating to next Gemini API key (index: {current_key_index})")

    # Get a new model with the next key
    return get_gemini_model()

# Function to rotate to the next model when one fails
def rotate_gemini_model():
    global current_model_index

    # Move to the next model
    current_model_index = (current_model_index + 1) % len(GEMINI_MODELS)
    logger.info(f"Rotating to next Gemini model: {GEMINI_MODELS[current_model_index]}")

    # Get a new model with the current key
    return get_gemini_model()

# Function to rotate both key and model
def rotate_gemini_key_and_model():
    global current_key_index, current_model_index

    # Move to the next key
    current_key_index += 1
    if current_key_index >= len(gemini_api_keys):
        current_key_index = 0

    # Move to the next model
    current_model_index = (current_model_index + 1) % len(GEMINI_MODELS)

    logger.info(f"Rotating to next Gemini key (index: {current_key_index}) and model: {GEMINI_MODELS[current_model_index]}")

    # Get a new model with the next key
    return get_gemini_model()

# Configure API clients
try:
    # Initialize with the first model in the list
    model = get_gemini_model(GEMINI_MODELS[0])
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")
    # Create a dummy model that will be replaced on first use
    model = None

serper_api_key = os.getenv("SERPER_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")

# Initialize Agno agent with Gemini model
def get_agno_agent(instructions=None, tools=None, model_name=None):
    """
    Create an Agno agent with specified Gemini model

    Args:
        instructions (str): Custom instructions for the agent
        tools (list): List of Agno tools to use with the agent
        model_name (str): Specific Gemini model to use (if None, uses current model in rotation)

    Returns:
        Agent: Configured Agno agent
    """
    default_instructions = dedent("""
        You are a helpful assistant for Tranquility Spa & Wellness Center.
        Provide accurate and helpful information about spa services, bookings, and policies.
        Be friendly, professional, and concise in your responses.
    """)

    # Get the current API key
    global current_key_index, current_model_index, gemini_api_keys
    api_key = gemini_api_keys[current_key_index]

    # Determine which model to use
    if model_name:
        model_id = model_name
    else:
        model_id = GEMINI_MODELS[current_model_index]

    logger.info(f"Creating Agno agent with model: {model_id}")

    # Create the agent with specified Gemini model
    agent = Agent(
        model=Gemini(id=model_id, api_key=api_key),
        instructions=instructions or default_instructions,
        tools=tools or []
    )

    return agent

# Get Agno agent with key and model rotation on failure
def get_agno_agent_with_retry(instructions=None, tools=None, model_name=None):
    """
    Create an Agno agent with Gemini model, with key and model rotation on failure

    Args:
        instructions (str): Custom instructions for the agent
        tools (list): List of Agno tools to use with the agent
        model_name (str): Specific Gemini model to use (if None, uses current model in rotation)

    Returns:
        Agent: Configured Agno agent
    """
    global current_key_index, current_model_index, gemini_api_keys

    max_retries = min(len(gemini_api_keys) * 2, len(gemini_api_keys) * len(GEMINI_MODELS))
    last_error = None

    for retry in range(max_retries):
        try:
            # Get a fresh API key and model for each attempt
            if retry > 0:
                # On odd retries, rotate key
                if retry % 2 == 1:
                    rotate_gemini_key()
                # On even retries, rotate model
                else:
                    rotate_gemini_model()
                # If we've tried multiple times, rotate both
                if retry >= 3:
                    rotate_gemini_key_and_model()

            # Create the agent with the current key and model
            agent = get_agno_agent(instructions, tools, model_name)

            # Don't test the agent - it might use up quota unnecessarily
            # Just return it and let it be used when needed
            return agent
        except Exception as e:
            last_error = e
            logger.error(f"Error creating Agno agent (attempt {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                logger.info(f"Retrying with next Gemini API key or model")

    # If we get here, all retries failed
    logger.error(f"All attempts to create Agno agent failed: {str(last_error)}")
    raise last_error
# Spa service business information
SPA_BUSINESS = {
    "name": "Tranquility Spa & Wellness Center",
    "description": "A premium spa offering massage therapy, facials, body treatments, and wellness services.",
    "hours": {
        "Monday": "9:00 AM - 8:00 PM",
        "Tuesday": "9:00 AM - 8:00 PM",
        "Wednesday": "9:00 AM - 8:00 PM",
        "Thursday": "9:00 AM - 8:00 PM",
        "Friday": "9:00 AM - 9:00 PM",
        "Saturday": "8:00 AM - 9:00 PM",
        "Sunday": "10:00 AM - 6:00 PM"
    },
    "services": [],
    "booking_policy": "Appointments must be booked at least 4 hours in advance. We recommend booking 2-3 days ahead for weekend appointments.",
    "cancellation_policy": "Cancellations must be made at least 24 hours before your appointment to avoid a 50% cancellation fee. No-shows will be charged the full service amount.",
    "contact": {
        "phone": "(555) 123-4567",
        "email": "appointments@tranquilityspa.com",
        "website": "www.tranquilityspa.com",
        "address": "123 Serenity Lane, Relaxville, CA 94123"
    }
}


# Database connection
def get_db_connection():
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    # Connect to the database
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        #print("Connection successful!")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Example query
        cursor.execute("SELECT * FROM public.services;")
        result = cursor.fetchall()
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Get services from database
def get_services_from_db():

    try:
        USER = os.getenv("user")
        PASSWORD = os.getenv("password")
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DBNAME = os.getenv("dbname")

        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        #print("Connection successful!")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Use the exact column names from the schema
        query = """
        SELECT "Service ID", "Service Name", "Description", "Price (INR)", "Category"
        FROM public.services;
        """

        cursor.execute(query)
        services = cursor.fetchall()

        result = []
        for service in services:
            result.append({
                "id": service[0] if service[0] is not None else "",
                "name": str(service[1]) if service[1] is not None else "",
                "description": str(service[2]) if service[2] is not None else "",
                "price": service[3] if service[3] is not None else "",
                "category": str(service[4]) if service[4] is not None else ""
            })

        cursor.close()
        connection.close()
        return result
    except Exception as e:
        logger.error(f"Error fetching services: {str(e)}")
        if connection:
            connection.close()
        return []

# Memory types
MEMORY_TYPES = {
    "INTERACTION": "interaction",           # Regular conversation
    "FAQ": "faq",                           # Frequently asked questions
    "BOOKING": "booking",                   # Booking information
    "PREFERENCE": "preference",             # User preferences
    "SERVICE_INTEREST": "service_interest"  # Services the user has shown interest in
}

# Store memory with classification
def store_memory(user_id, message, response, memory_type=MEMORY_TYPES["INTERACTION"], additional_metadata=None):
    # Ensure response is not None
    if response is None:
        response = "No response generated"

    combined = f"User: {message}\nBot: {response}"
    vector = get_embedding(combined)
    timestamp = int(time.time())
    memory_id = f"{user_id}-{uuid.uuid4()}"

    metadata = {
        "user_id": user_id,
        "message": message,
        "response": response,
        "timestamp": timestamp,
        "type": memory_type
    }

    # Add any additional metadata
    if additional_metadata:
        # Ensure all metadata values are valid types for Pinecone
        sanitized_metadata = {}
        for key, value in additional_metadata.items():
            if value is None:
                sanitized_metadata[key] = "None"
            elif isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(item, str) for item in value)):
                sanitized_metadata[key] = value
            else:
                sanitized_metadata[key] = str(value)

        metadata.update(sanitized_metadata)

    index.upsert([
        (memory_id, vector, metadata)
    ])

    # If this is a FAQ, also store it separately
    if memory_type == MEMORY_TYPES["FAQ"]:
        # Extract the question part from the message
        question = extract_question(message)
        if question:
            faq_vector = get_embedding(question)
            faq_id = f"{user_id}-faq-{uuid.uuid4()}"
            index.upsert([
                (faq_id, faq_vector, {
                    "user_id": user_id,
                    "question": question,
                    "answer": response,
                    "timestamp": timestamp,
                    "type": "faq_entry"
                })
            ])

def extract_question(text):
    """Extract a question from text if present"""
    # Simple pattern matching for questions
    question_patterns = [
        r'(?:^|\s)(?:what|how|when|where|why|who|can|could|would|will|is|are|do|does|did|should)(?:\s+\w+){2,}[?]',
        r'(?:^|\s)(?:tell me about|explain|describe)(?:\s+\w+){2,}[?]?',
        r'[^.!?]*[?]'
    ]

    for pattern in question_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()

    return None

# Recall semantic memory by filtering matches based on user_id
def recall_semantic_memory(user_id, query, top_k=3, memory_types=None):
    query_vec = get_embedding(query)

    # Build filter
    filter_dict = {"user_id": {"$eq": user_id}}

    # Add memory type filter if specified
    if memory_types:
        if isinstance(memory_types, list):
            filter_dict["type"] = {"$in": memory_types}
        else:
            filter_dict["type"] = {"$eq": memory_types}

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    return res.get("matches", [])

# Get recent conversation history for context
def get_conversation_history(user_id, limit=5):
    # Query the most recent interactions for this user
    res = index.query(
        vector=[0] * 768,  # Dummy vector, we're just using filters (Gemini dimension)
        top_k=limit * 2,  # Get more than needed to filter
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": MEMORY_TYPES["INTERACTION"]}
        }
    )

    # Sort by timestamp (newest first)
    matches = res.get("matches", [])
    sorted_matches = sorted(matches, key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)

    return sorted_matches[:limit]

# Get FAQs for this user
def get_user_faqs(user_id, limit=5):
    res = index.query(
        vector=[0] * 768,  # Dummy vector, we're just using filters (Gemini dimension)
        top_k=limit,
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": "faq_entry"}
        }
    )

    return res.get("matches", [])

# This function is no longer used - we'll use SQLTools with Agno instead
# Keeping the function signature for compatibility
def get_bookings_from_db(booking_type="all", customer_name=None, future_days=30, past_days=30):
    """
    This function is deprecated. We now use SQLTools with Agno instead.
    """
    logger.warning("get_bookings_from_db is deprecated. Use SQLTools with Agno instead.")
    return []

# Get user preferences
def get_user_preferences(user_id):
    res = index.query(
        vector=[0] * 768,  # Dummy vector (Gemini dimension)
        top_k=10,
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": MEMORY_TYPES["PREFERENCE"]}
        }
    )

    preferences = {}
    for match in res.get("matches", []):
        metadata = match.get("metadata", {})
        if "preference_key" in metadata and "preference_value" in metadata:
            preferences[metadata["preference_key"]] = metadata["preference_value"]

    return preferences

# Store user preference
def store_user_preference(user_id, key, value):
    # Check if preference already exists
    res = index.query(
        vector=[0] * 768,  # Gemini dimension
        top_k=1,
        include_metadata=True,
        filter={
            "user_id": {"$eq": user_id},
            "type": {"$eq": MEMORY_TYPES["PREFERENCE"]},
            "preference_key": {"$eq": key}
        }
    )

    matches = res.get("matches", [])
    timestamp = int(time.time())

    if matches:
        # Update existing preference
        memory_id = matches[0]["id"]
        vector = get_embedding(f"{key}: {value}")

        index.upsert([
            (memory_id, vector, {
                "user_id": user_id,
                "preference_key": key,
                "preference_value": value,
                "timestamp": timestamp,
                "type": MEMORY_TYPES["PREFERENCE"]
            })
        ])
    else:
        # Create new preference
        memory_id = f"{user_id}-pref-{uuid.uuid4()}"
        vector = get_embedding(f"{key}: {value}")

        index.upsert([
            (memory_id, vector, {
                "user_id": user_id,
                "preference_key": key,
                "preference_value": value,
                "timestamp": timestamp,
                "type": MEMORY_TYPES["PREFERENCE"]
            })
        ])

# Format conversation history into a readable context
def format_conversation_context(history):
    # Reverse to get chronological order (oldest first)
    history = list(reversed(history))
    context = []
    for item in history:
        metadata = item['metadata']
        context.append(f"User: {metadata['message']}")
        context.append(f"Assistant: {metadata['response']}")

    return "\n".join(context)

# Search the web for additional information
def search_web(query, num_results=3):
    try:
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        payload = json.dumps({
            "q": query,
            "num": num_results
        })

        response = requests.post('https://google.serper.dev/search', headers=headers, data=payload)

        if response.status_code == 200:
            results = response.json()
            organic_results = results.get('organic', [])

            formatted_results = []
            for result in organic_results:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                link = result.get('link', '')
                formatted_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n")

            return "\n".join(formatted_results)
        else:
            logger.error(f"Web search failed with status code {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return ""

# Extract content from a webpage
def extract_webpage_content(url):
    try:
        headers = {
            'Authorization': f'Bearer {jina_api_key}'
        }

        response = requests.post(
            'https://api.jina.ai/v1/reader',
            json={"url": url},
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('text', '')
        else:
            logger.error(f"Webpage extraction failed with status code {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error during webpage extraction: {str(e)}")
        return ""

# Analyze user message for intent and entities
def analyze_message(message):
    # Define patterns for different intents
    booking_patterns = [
        r'book(?:ing)?', r'appoint(?:ment)?', r'schedule', r'reserve',
        r'available', r'time(?:s)?', r'slot(?:s)?', r'when can I',
        r'make a booking', r'book a', r'confirm', r'reservation'
    ]

    pricing_patterns = [
        r'price(?:s)?', r'cost(?:s)?', r'how much', r'fee(?:s)?',
        r'rate(?:s)?', r'pricing', r'dollar(?:s)?', r'\$', r'₹', r'inr'
    ]

    service_patterns = [
        r'service(?:s)?', r'massage', r'facial', r'treatment(?:s)?',
        r'package(?:s)?', r'offer(?:ing)?', r'spa', r'therapy'
    ]

    policy_patterns = [
        r'policy', r'policies', r'cancel(?:lation)?', r'refund',
        r'reschedule', r'change', r'booking policy'
    ]

    # Check for intents
    intents = []

    if any(re.search(pattern, message, re.IGNORECASE) for pattern in booking_patterns):
        intents.append("booking")

    if any(re.search(pattern, message, re.IGNORECASE) for pattern in pricing_patterns):
        intents.append("pricing")

    if any(re.search(pattern, message, re.IGNORECASE) for pattern in service_patterns):
        intents.append("service_info")

    if any(re.search(pattern, message, re.IGNORECASE) for pattern in policy_patterns):
        intents.append("policy")

    # Extract service names if mentioned
    service_entities = []
    # Get services from database for entity extraction
    try:
        services = get_services_from_db()
        if services:
            for service in services:
                # Convert to string to ensure we can call lower() and handle None values
                service_name = str(service["name"]).lower() if service["name"] is not None else ""
                if service_name and service_name in message.lower():
                    service_entities.append(str(service["name"]))
    except Exception as e:
        logger.error(f"Error extracting service entities: {str(e)}")

    # Check for date entities
    date_entities = []
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
        r'(tomorrow|next\s+\w+day|this\s+\w+day|\w+day)'  # Natural language
    ]

    for pattern in date_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            date_entities.append(match.group(1))

    # Check for month and day mentions (like "april 7")
    month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})'
    month_match = re.search(month_pattern, message, re.IGNORECASE)
    if month_match:
        month_name = month_match.group(1).lower()
        day = int(month_match.group(2))

        # Convert month name to number
        month_dict = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        month_num = month_dict.get(month_name)
        if month_num:
            # Always use 2025 for bookings
            year = 2025

            date_str = f"{year}-{month_num:02d}-{day:02d}"
            date_entities.append(date_str)

    # Check for time entities
    time_entities = []
    time_pattern = r'(\d{1,2}:\d{2}\s*[AP]M|\d{1,2}\s*[AP]M|\d{1,2}[AP]M)'

    for match in re.finditer(time_pattern, message, re.IGNORECASE):
        time_entities.append(match.group(1))

    return {
        "intents": intents,
        "service_entities": service_entities,
        "date_entities": date_entities,
        "time_entities": time_entities
    }

# Process user message and generate response
def process_message(user_id, message):
    # Analyze message for intents and entities
    analysis = analyze_message(message)
    intents = analysis["intents"]
    service_entities = analysis["service_entities"]
    date_entities = analysis.get("date_entities", [])
    time_entities = analysis.get("time_entities", [])

    # Handle "who am I" or similar identity queries
    if message.lower().strip() in ["who am i", "who am i?", "tell me about myself"]:
        preferences = get_user_preferences(user_id)
        semantic_memories = recall_semantic_memory(user_id, "user identity", top_k=1)
        
        response_text = f"You’re user {user_id} at Tranquility Spa & Wellness Center! "
        if preferences:
            response_text += "I know you prefer: " + ", ".join([f"{k}: {v}" for k, v in preferences.items()]) + ". "
        if semantic_memories:
            last_meta = semantic_memories[0]['metadata']
            last_user_text = last_meta.get('message', last_meta.get('question', ''))
            if last_user_text:
                response_text += f"Last time, you asked about '{last_user_text}'. "
        response_text += "How can I assist you today?"
        
        store_memory(user_id, message, response_text)
        return response_text

    # Store service interests if detected
    for service in service_entities:
        store_memory(
            user_id,
            message,
            f"User showed interest in {service}",
            MEMORY_TYPES["SERVICE_INTEREST"],
            {"service_name": service}
        )

    # Get recent conversation history
    conversation_history = get_conversation_history(user_id)
    conversation_context = format_conversation_context(conversation_history)

    # Get semantically similar memories
    semantic_memories = recall_semantic_memory(user_id, message)
    semantic_context = ""
    if semantic_memories:
        context_lines = []
        for m in semantic_memories:
            metadata = m.get('metadata', {})
            user_text = metadata.get('message', metadata.get('question', ''))
            bot_text = metadata.get('response', metadata.get('answer', ''))
            if user_text and bot_text:
                context_lines.append(f"Related memory: User: {user_text}\nAssistant: {bot_text}")
        semantic_context = "\n\n".join(context_lines) if context_lines else "No relevant memories found."

    # Get user FAQs
    user_faqs = get_user_faqs(user_id)
    faq_context = "\n\n".join([
        f"FAQ: {m['metadata']['question']}\nAnswer: {m['metadata']['answer']}"
        for m in user_faqs
    ])

    # Get user preferences
    preferences = get_user_preferences(user_id)
    preference_context = "\n".join([f"{key}: {value}" for key, value in preferences.items()])

    # Prepare business information based on intents
    business_info = f"Business Name: {SPA_BUSINESS['name']}\nDescription: {SPA_BUSINESS['description']}\n"

    if "booking" in intents:
        business_info += f"\nBooking Policy: {SPA_BUSINESS['booking_policy']}\n"

        # Use the simulated data for now, only use agno for actual booking creation
        available_slots = {}
        today = datetime.now()

        for i in range(7):  # 7 days ahead
            date = today + timedelta(days=i)
            day_name = date.strftime("%A")
            date_str = date.strftime("%Y-%m-%d")

            # Skip if business is closed that day
            if day_name not in SPA_BUSINESS["hours"]:
                continue

            # Parse business hours
            hours = SPA_BUSINESS["hours"][day_name]
            start_time, end_time = hours.split(" - ")

            # Convert to datetime objects
            start_dt = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %I:%M %p")
            end_dt = datetime.strptime(f"{date_str} {end_time}", "%Y-%m-%d %I:%M %p")

            # Generate time slots every 30 minutes
            slots = []
            current = start_dt
            while current < end_dt:
                # Simulate some slots being already booked
                if hash(current.strftime("%Y-%m-%d %H:%M")) % 3 != 0:  # 1/3 of slots are booked
                    slots.append(current.strftime("%I:%M %p"))
                current += timedelta(minutes=30)

            if slots:
                available_slots[date_str] = slots

        if service_entities:
            service_name = service_entities[0]
            business_info += f"Available time slots for {service_name} in the next 7 days:\n"
        else:
            business_info += f"Available time slots for the next 7 days:\n"

        for date, slots in available_slots.items():
            business_info += f"{date}: {', '.join(slots[:5])}"
            if len(slots) > 5:
                business_info += f" and {len(slots) - 5} more slots"
            business_info += "\n"

    if "pricing" in intents or "service_info" in intents:
        # Initialize agno agent for service information
        try:
            db_url = f"postgresql+psycopg://{os.getenv('user1')}:{os.getenv('password1')}@{os.getenv('host1')}:{os.getenv('port1')}/{os.getenv('dbname1')}"

            # Create Agno agent with Gemini 1.5 Pro and SQL tools
            sql_tools = SQLTools(db_url=db_url)
            agent_instructions = dedent("""
                You are a spa service information specialist for Tranquility Spa & Wellness Center.
                Provide detailed and accurate information about our services, including prices, durations, and descriptions.
                Format the information in a clear, easy-to-read manner.
                If specific services are requested, focus on providing details about those services.

                DATABASE SCHEMA (EXACT NAMES - DO NOT MODIFY):
                - Table name: public.services (not spa_services)
                  Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"
                  Note: Column names include spaces and must be quoted in SQL queries

                EXAMPLE CORRECT SQL QUERIES:
                - SELECT * FROM public.services WHERE "Service Name" = 'Glowing Facial'
                - SELECT * FROM public.services WHERE "Category" = 'Facial'
            """)
            # Create agent and query for services
            agent = None
            try:
                agent = get_agno_agent_with_retry(instructions=agent_instructions, tools=[sql_tools])
            except Exception as e:
                logger.error(f"Failed to create Agno agent after retries: {str(e)}")
                business_info += "I'm sorry, but I couldn't retrieve the service information at this moment. Please try again later or contact us directly for service details."

            # Only proceed if we successfully created an agent
            if agent:
                # Query for services
                agent_prompt = "List all services available at the spa with their descriptions, durations, and prices."
                if service_entities:
                    service_names = ", ".join(service_entities)
                    agent_prompt = f"Find information about these spa services: {service_names}."

                business_info += "\nServices:\n"
                try:
                    response = agent.run(agent_prompt)
                    agent_response = response.content

                    if agent_response and "error" not in agent_response.lower():
                        business_info += agent_response
                    else:
                        business_info += "I'm sorry, but I couldn't retrieve the service information at this moment. Please try again later or contact us directly for service details."
                except Exception as e:
                    logger.error(f"Error getting service information from agent: {str(e)}")
                    business_info += "I'm sorry, but I couldn't retrieve the service information at this moment. Please try again later or contact us directly for service details."
        except Exception as e:
            logger.error(f"Error retrieving service information: {str(e)}")
            business_info += "I'm sorry, but I couldn't retrieve the service information at this moment. Please try again later or contact us directly for service details."

    if "policy" in intents:
        business_info += f"\nCancellation Policy: {SPA_BUSINESS['cancellation_policy']}\n"

    # Combine all context
    context = "You are a helpful assistant for Tranquility Spa & Wellness Center. Use the following information to answer the user's question.\n\n"
    context += f"BUSINESS INFORMATION:\n{business_info}\n\n"
    if preference_context:
        context += f"USER PREFERENCES:\n{preference_context}\n\n"
    if conversation_context:
        context += f"RECENT CONVERSATION HISTORY:\n{conversation_context}\n\n"
    if semantic_context:
        context += f"RELEVANT MEMORIES:\n{semantic_context}\n\n"
    if faq_context:
        context += f"FREQUENTLY ASKED QUESTIONS:\n{faq_context}\n\n"

    # Create the final prompt
    prompt = f"{context}\nUser: {message}\n\nAssistant:"

    # Generate response using Gemini with key rotation on failure
    max_retries = min(3, len(gemini_api_keys))
    response_text = None

    for retry in range(max_retries):
        try:
            # Get a fresh model instance for each attempt
            current_model = get_gemini_model() if retry > 0 else model

            # Try to generate content
            response = current_model.generate_content(prompt)
            response_text = response.text
            # If successful, break out of the loop
            break
        except Exception as e:
            logger.error(f"Error generating content with Gemini (attempt {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                # Rotate to the next key and try again
                rotate_gemini_key()
                logger.info(f"Retrying with next Gemini API key")
            else:
                # All retries failed, set a fallback response
                response_text = "I'm sorry, but I'm having trouble generating a response right now. Please try again later."

    # If we somehow didn't get a response_text (should never happen due to fallback)
    if response_text is None:
        response_text = "I'm sorry, but I'm having trouble generating a response right now. Please try again later."

    # Check if this is a booking-related message
    booking_request = False

    # Check if the message contains booking intent or mentions booking
    if "booking" in intents or any(word in message.lower() for word in ["book", "schedule", "reserve", "make appointment", "booking", "slot", "appointment"]):
        # Handle booking with Agno
        booking_response = handle_booking_with_agno(
            user_id,
            message,
            intents,
            service_entities,
            date_entities,
            time_entities
        )

        # Replace the generated response with the booking response if we got one
        if booking_response:
            response_text = booking_response

            # If booking was confirmed, mark as booking request
            if "confirmed" in booking_response.lower() or "booked" in booking_response.lower():
                booking_request = True
        # If booking_response is None, keep the original response_text

    # Check if this is a FAQ
    if extract_question(message):
        is_faq = any(intent in intents for intent in ["pricing", "service_info", "policy", "booking"])
        memory_type = MEMORY_TYPES["FAQ"] if is_faq else MEMORY_TYPES["INTERACTION"]
        store_memory(user_id, message, response_text, memory_type)
    else:
        store_memory(user_id, message, response_text)

    # If this was a booking request, store it as booking memory
    if booking_request:
        # Extract service name from the response for metadata
        service_match = None
        try:
            services = get_services_from_db()
            if services:
                for service in services:
                    # Convert to string to ensure we can call lower()
                    service_name = str(service["name"]).lower()
                    if service_name in response_text.lower():
                        service_match = str(service["name"])
                        break
        except Exception as e:
            logger.error(f"Error extracting service from booking: {str(e)}")

        additional_metadata = {"service_booked": service_match} if service_match else None
        store_memory(user_id, message, response_text, MEMORY_TYPES["BOOKING"], additional_metadata)

    # Extract and store preferences in a separate thread
    threading.Thread(target=extract_and_store_preferences, args=(user_id, message, response_text)).start()

    return response_text

def extract_and_store_preferences(user_id, message, response):
    """Extract and store user preferences from the conversation"""
    # Look for service preferences from database
    try:
        services = get_services_from_db()
        if services:
            for service in services:
                # Convert to string to ensure we can call lower() and handle None values
                service_name = str(service["name"]).lower() if service["name"] is not None else ""
                if service_name and service_name in message.lower() and any(word in message.lower() for word in ["like", "prefer", "favorite", "enjoy"]):
                    store_user_preference(user_id, "preferred_service", str(service["name"]))
    except Exception as e:
        logger.error(f"Error extracting preferences: {str(e)}")

    # Look for time preferences
    time_patterns = [
        (r'prefer\s+(\w+day)', "preferred_day"),
        (r'prefer\s+(morning|afternoon|evening)', "preferred_time"),
        (r'(morning|afternoon|evening)\s+is better', "preferred_time")
    ]

    for pattern, key in time_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            store_user_preference(user_id, key, match.group(1))

# Handle booking with Agno
def handle_booking_with_agno(user_id, message, intents, service_entities, date_entities=None, time_entities=None):
    """
    Handle booking using Agno agent

    Args:
        user_id (str): User ID
        message (str): User message
        intents (list): Detected intents
        service_entities (list): Detected service entities
        date_entities (list, optional): Detected date entities
        time_entities (list, optional): Detected time entities

    Returns:
        str: Response message from Agno agent
    """
    try:
        import os
        from textwrap import dedent
        # Assume SQLTools, get_agno_agent_with_retry, get_agno_agent, rotate_gemini_key, logger, gemini_api_keys are imported or defined elsewhere in your code.

        # Create database connection string
        db_url = f"postgresql+psycopg://{os.getenv('user1')}:{os.getenv('password1')}@{os.getenv('host1')}:{os.getenv('port1')}/{os.getenv('dbname1')}"
        
        # Create SQL tools for Agno
        sql_tools = SQLTools(db_url=db_url)
        
        # Create booking agent with specific instructions
        booking_instructions = dedent("""
            You are a booking assistant for Tranquility Spa & Wellness Center.

            IMPORTANT BOOKING RULES:
            1. Bookings must be made at least 4 hours in advance
            2. Available time slots are from 9:00 AM to 7:00 PM, every hour
            3. Each service has a 1 hour duration and price
            4. The user must provide: service name, date, and time
            5. Check if the requested time slot is available before confirming by checking bookings table for any existing booking of any service in that particular day and time slot
            6. CRITICAL: When working with dates in SQL queries, ALWAYS use the PostgreSQL date format 'YYYY-MM-DD'
               - CORRECT: WHERE "Booking Date" = '2025-05-23'
               - INCORRECT: WHERE "Booking Date" = 'may 23'
            6. If no year mentioned, always use 2025 for all bookings
            DATABASE SCHEMA (EXACT NAMES - DO NOT MODIFY):
            - Table name: public.services (not spa_services)
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"
              Note: Column names include spaces and must be quoted in SQL queries

            - Table name: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: Column names include spaces and must be quoted in SQL queries

            EXAMPLE CORRECT SQL QUERIES:
            - SELECT * FROM public.services WHERE "Service Name" = 'Glowing Facial'
            - SELECT * FROM public.bookings WHERE "Booking Date" = '2025-04-07'
            - SELECT "Service ID", "Service Name", "Price (INR)" FROM public.services WHERE "Service ID" = 1
            - INSERT INTO public.bookings ("Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)")
              VALUES ('user123', 1, '2025-04-07', '14:00', 1500.00)

            BOOKING PROCESS:
            1. Ask for any missing information (service, date, time) unless the user provides a direct booking request.
            2. Use the user's ID as the "Customer Name" when creating bookings.
            3. Check if the service exists in the database using the EXACT table and column names then use Service Name to get Service ID.
            4. IMPORTANT: Fetch the price from the services database using the Service ID - use the "Price (INR)" column value.
            5. Check if the requested time slot is available.
            6. Create the booking in the database using the user's ID as the "Customer Name" and include the price fetched from the services table.
            7. Confirm the booking details to the user including the price.
            
            In case of a direct booking request (for example, if the user says "can you book"), process immediately and return a success confirmation message formatted as:
            "Your booking for <Service Name> on <Date> at <Time> has been successfully created. The total price is INR <price>."
            
            Always be helpful, professional, and provide clear instructions.
        """)
        
        # Create the Agno agent with key rotation on failure
        try:
            booking_agent = get_agno_agent_with_retry(instructions=booking_instructions, tools=[sql_tools])
        except Exception as e:
            logger.error(f"Failed to create booking agent after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble with the booking system right now. Please try again later or contact us directly at (555) 123-4567 to make a booking."

        # Prepare the prompt for the booking agent
        booking_prompt = f"User wants to book a spa service. The user's ID is '{user_id}' - use this as the 'Customer Name' when creating the booking. "
        if service_entities:
            service_names = ", ".join(service_entities)
            booking_prompt += f"They mentioned these services: {service_names}. "
        if date_entities:
            dates = ", ".join(date_entities)
            booking_prompt += f"They mentioned these dates: {dates}. "
        if time_entities:
            times = ", ".join(time_entities)
            booking_prompt += f"They mentioned these times: {times}. "
        booking_prompt += f"Here's what they said: '{message}'. Handle this booking request appropriately."

        # Check if the user's message is a direct booking instruction.
        direct_booking = "can you book" in message.lower()

        # If the request is direct and required information is presumed to be provided,
        # then we bypass additional clarifications and directly query the database.
        if direct_booking:
            # For this example, we assume the service, date, and time are exactly as follows.
            # In a real-world scenario, you might extract these details using entity recognition.
            service_name = service_entities[0]
            booking_date = date_entities[0]
            booking_time = time_entities[0]
            # Build SQL query to get the service details (price and Service ID)
            service_query = f"SELECT \"Service ID\", \"Price (INR)\" FROM public.services WHERE \"Service Name\" = '{service_name}'"
            service_result = json.loads(sql_tools.run_sql_query(service_query))  # Use run_sql_query instead of execute

            if not service_result:
                return f"Sorry, the service '{service_name}' is not available."
            
            # For simplicity, assume service_result is a dict-like object or a row with keys "Service ID" and "Price (INR)"
            service_id = service_result[0]["Service ID"]
            price = service_result[0]["Price (INR)"]

            # Build SQL query to check if the time slot is available
            booking_check_query = f"SELECT * FROM public.bookings WHERE \"Booking Date\" = '{booking_date}' AND \"Time Slot (HH:MM)\" = '{booking_time}'"
            booking_check_result = json.loads(sql_tools.run_sql_query(booking_check_query))  # Use run_sql_query instead of execute
            if booking_check_result:
                return f"Sorry, the time slot {booking_time} on {booking_date} is already booked. Please choose another time."

            # Build SQL query to insert the booking
            booking_insert_query = (
                f"INSERT INTO public.bookings (\"Customer Name\", \"Service ID\", \"Booking Date\", \"Time Slot (HH:MM)\", \"Price (INR)\") "
                f"VALUES ('{user_id}', {service_id}, '{booking_date}', '{booking_time}', {price})"
            )
            # For INSERT queries, we still use run_sql_query but need to handle the result differently
            sql_tools.run_sql_query(booking_insert_query)
            # Since INSERT doesn't return rows, we can consider it successful if no exception was raised
            insertion_result = True
            if insertion_result:  # Since we set insertion_result to True if no exception was raised
                return f"Your booking for {service_name} on {booking_date} at {booking_time} has been successfully created. The total price is INR {price}."
            else:
                return "I'm sorry, there was an error processing your booking. Please try again later."

        # If not a direct booking request, use the agent logic with retry.
        max_retries = min(2, len(gemini_api_keys))
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key
                if retry > 0:
                    logger.info(f"Retrying booking agent with new API key (attempt {retry+1})")
                    rotate_gemini_key()
                    booking_agent = get_agno_agent(instructions=booking_instructions, tools=[sql_tools])

                response = booking_agent.run(booking_prompt)
                agent_response = response.content

                if agent_response is None:
                    raise ValueError("Agno agent returned None response")

                logger.info(f"Booking interaction - User: {user_id}, Message: {message}, Response: {agent_response[:100]}...")
                return agent_response
            except Exception as inner_e:
                logger.error(f"Error getting response from Agno agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    continue
                else:
                    return "I'd be happy to help you book a spa service. Could you please tell me which service you're interested in, what date, and what time you prefer? Alternatively, you can contact us directly at (555) 123-4567 to make a booking."
    except Exception as e:
        logger.error(f"Error handling booking with Agno: {str(e)}")
        return "I'm sorry, but I'm having trouble with the booking system right now. Please try again later or contact us directly at (555) 123-4567 to make a booking."

# Classify booking message type using Agno and Gemini
def classify_booking_message(message):
    """
    Classify a message as either "booking_retrieval" or "create_booking" using Gemini with Agno

    Args:
        message (str): User message to classify

    Returns:
        str: Classification result - either "booking_retrieval" or "create_booking"
    """
    try:
        # Create instructions for the classification agent
        classification_instructions = dedent("""
            You are a message classification assistant for Tranquility Spa & Wellness Center.
            Your task is to analyze user messages and classify them into one of two categories:
            
            1. "booking_retrieval" - Messages asking about existing bookings, checking booking status, 
               or requesting information about past or upcoming appointments.
               
            2. "create_booking" - Messages expressing intent to make a new booking, schedule an appointment,
               or reserve a service.
            
            CLASSIFICATION GUIDELINES:
            - "booking_retrieval" includes: checking appointment status, viewing upcoming bookings, 
              asking about past bookings, requesting booking details, etc.
              
            - "create_booking" includes: making new appointments, scheduling services, requesting available 
              time slots with intent to book, etc.
              
            RESPONSE FORMAT:
            Respond with ONLY ONE of these two values:
            - "booking_retrieval"
            - "create_booking"
            
            Do not include any other text, explanation, or formatting in your response.
        """)
        
        # Create the Agno agent with key rotation on failure
        try:
            classification_agent = get_agno_agent_with_retry(instructions=classification_instructions)
        except Exception as e:
            logger.error(f"Failed to create classification agent after retries: {str(e)}")
            # Default to create_booking if we can't classify
            return "create_booking"
        
        # Prepare the prompt for the classification agent
        classification_prompt = f"Classify this message as either 'booking_retrieval' or 'create_booking': '{message}'"
        
        # Get response from Agno agent with retry
        max_retries = min(2, len(gemini_api_keys))
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key
                if retry > 0:
                    logger.info(f"Retrying classification agent with new API key (attempt {retry+1})")
                    rotate_gemini_key()
                    classification_agent = get_agno_agent(instructions=classification_instructions)
                
                # Try to get a response
                response = classification_agent.run(classification_prompt)
                agent_response = response.content.strip().lower()
                
                # Make sure we have a valid response
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")
                
                # Normalize the response to ensure it's one of our expected values
                if "retrieval" in agent_response:
                    return "booking_retrieval"
                elif "create" in agent_response:
                    return "create_booking"
                else:
                    # If the response doesn't clearly match either category, default to create_booking
                    logger.warning(f"Unclear classification response: {agent_response}. Defaulting to create_booking.")
                    return "create_booking"
                
            except Exception as inner_e:
                logger.error(f"Error getting response from classification agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    # Will retry with a new key
                    continue
                else:
                    # All retries failed, default to create_booking
                    logger.error("All classification attempts failed. Defaulting to create_booking.")
                    return "create_booking"
    
    except Exception as e:
        logger.error(f"Error classifying booking message: {str(e)}")
        # Default to create_booking if we encounter an error
        return "create_booking"

# Format date string to PostgreSQL compatible format
def format_date_for_postgresql(date_str):
    """
    Convert various date formats to PostgreSQL compatible 'YYYY-MM-DD' format
    
    Args:
        date_str (str): Date string in various formats (e.g., 'may 23', 'April 15')
        
    Returns:
        str: Date string in 'YYYY-MM-DD' format or original string if conversion fails
    """
    if not date_str:
        return date_str
        
    # If already in YYYY-MM-DD format, return as is
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
        
    try:
        # Try different date formats
        formats_to_try = [
            # Month name formats
            '%b %d',      # May 23
            '%B %d',      # May 23
            '%b %d %Y',   # May 23 2025
            '%B %d %Y',   # May 23 2025
            '%b %d, %Y',  # May 23, 2025
            '%B %d, %Y',  # May 23, 2025
            
            # Day first formats
            '%d %b %Y',   # 23 May 2025
            '%d %B %Y',   # 23 May 2025
            '%d %b',      # 23 May
            '%d %B',      # 23 May
            
            # Numeric formats
            '%m/%d/%Y',   # 05/23/2025
            '%m-%d-%Y',   # 05-23-2025
            '%m/%d',      # 05/23
            '%m-%d',      # 05-23
            
            # ISO formats
            '%Y/%m/%d',   # 2025/05/23
            '%Y-%m-%d',   # 2025-05-23
            '%Y-%m',      # 2025-05
        ]
        
        # Special case for "April 23 2024" format
        if re.match(r'^[A-Za-z]+\s+\d{1,2}\s+\d{4}$', date_str):
            try:
                date_obj = datetime.strptime(date_str, '%B %d %Y')
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, '%b %d %Y')
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    pass  # Continue with other formats
        
        # Default year to use if not specified
        default_year = 2025
        current_date = datetime.now()
        
        for fmt in formats_to_try:
            try:
                # Try parsing with the current format
                if '%Y' in fmt:
                    # Format includes year
                    date_obj = datetime.strptime(date_str, fmt)
                else:
                    # Format doesn't include year, add default year
                    date_with_year = f"{date_str} {default_year}"
                    date_obj = datetime.strptime(date_with_year, f"{fmt} %Y")
                    
                    # If the resulting date is in the past, use next year
                    if date_obj.replace(year=current_date.year) < current_date:
                        date_obj = date_obj.replace(year=current_date.year + 1)
                    else:
                        date_obj = date_obj.replace(year=current_date.year)
                
                # Return formatted date
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                # Try next format
                continue
                
        # If all formats fail, try a more flexible approach with dateutil
        try:
            from dateutil import parser
            date_obj = parser.parse(date_str, fuzzy=True)
            
            # If no year was specified, use default year
            if date_obj.year == current_date.year and date_str.lower().find(str(current_date.year)) == -1:
                # If the resulting date is in the past, use next year
                if date_obj < current_date:
                    date_obj = date_obj.replace(year=default_year)
                    
            return date_obj.strftime('%Y-%m-%d')
        except:
            # If all parsing attempts fail, return original string
            logger.warning(f"Could not parse date string: {date_str}")
            return date_str
            
    except Exception as e:
        logger.error(f"Error formatting date '{date_str}': {str(e)}")
        return date_str

def handle_booking_query_with_agno(user_id, message, query_type="all", customer_name=None):
    """
    Handle booking query using Agno agent with SQLTools.
    The agent will analyze the user's request, craft the appropriate SQL date filters, 
    and generate a response using the user_id as the customer_name.
    
    Args:
        user_id (str): User ID.
        message (str): User message.
        query_type (str): Type of bookings to query - "previous", "future", or "all".
        customer_name (str): Optional customer name to filter bookings.
    
    Returns:
        str: Response message from Agno agent.
    """
    try:
        # Build the database URL
        db_url = f"postgresql+psycopg://{os.getenv('user')}:{os.getenv('password')}@" \
                 f"{os.getenv('host')}:{os.getenv('port')}/{os.getenv('dbname')}"
        
        # Instantiate SQLTools with the given db_url
        sql_tools = SQLTools(db_url=db_url)
        
        # Helper: Function to format dates for PostgreSQL (if needed)
        def format_sql_query_dates(query):
            """Ensure dates in SQL queries are in the correct YYYY-MM-DD format."""
            date_pattern = r'"Booking Date"\s*=\s*\'([^\']+)\''
            for match in re.finditer(date_pattern, query):
                date_str = match.group(1)
                if not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    formatted_date = format_date_for_postgresql(date_str)
                    query = query.replace(f'"Booking Date" = \'{date_str}\'',
                                          f'"Booking Date" = \'{formatted_date}\'')
            return query

        # Extract date filter information from the user's message.
        # For example, your extract_date_filter function might return a dictionary like:
        # { 'relation': 'before', 'date': '2024-04-20' }
        # It should recognize other relations such as 'after', 'on or after', 'on or before',
        # or even month-based filters (with relation 'month').
        date_filter = extract_date_filter(message)

        # Create the instructions for the Agno agent. This prompt now emphasizes that
        # the agent should analyze the user's request to apply any date filters dynamically.
        booking_query_instructions = dedent("""
            You are a booking assistant for Tranquility Spa & Wellness Center.
            Your job is to analyze the user's natural language request and craft an SQL query that uses
            any required date filters while ensuring that results are filtered using the user id as the
            "Customer Name" column.

            DATABASE SCHEMA (DO NOT MODIFY):
            - Table: public.services
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"

            - Table: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: The "Customer Name" column contains the user ID.
              Note: The "Booking Date" column is of DATE type and must be in 'YYYY-MM-DD' format.

            QUERY REQUIREMENTS:
            1. Always join public.bookings and public.services on "Service ID" to obtain the "Service Name".
            2. Always filter by "Customer Name" using the provided user id.
            3. If the user request contains a date filter, analyze the nature of the filter:
                - It might be "before" a given date,
                - "after" a given date,
                - "on or before" or "on or after" a specific date,
                - or for a particular month (use TO_CHAR(b."Booking Date", 'YYYY-MM') = 'YYYY-MM').
            4. If the query type is "future", always include:
                b."Booking Date" >= CURRENT_DATE
               to ensure only upcoming bookings are included.
            5. Use PostgreSQL date functions TO_DATE and TO_CHAR as necessary.
            6. Do not assume the date is in natural language; convert any date values to 'YYYY-MM-DD'.

            EXAMPLES:
            - For previous bookings:
              SELECT b."Booking ID", b."Customer Name", s."Service Name",
                     b."Booking Date", b."Time Slot (HH:MM)", b."Price (INR)"
              FROM public.bookings b
              JOIN public.services s ON b."Service ID" = s."Service ID"
              WHERE b."Customer Name" = 'user_id' AND b."Booking Date" < CURRENT_DATE
              ORDER BY b."Booking Date" DESC, b."Time Slot (HH:MM)"
              
            - For future bookings with a specific upper date limit:
              SELECT b."Booking ID", b."Customer Name", s."Service Name",
                     b."Booking Date", b."Time Slot (HH:MM)", b."Price (INR)"
              FROM public.bookings b
              JOIN public.services s ON b."Service ID" = s."Service ID"
              WHERE b."Customer Name" = 'user_id'
                AND b."Booking Date" >= CURRENT_DATE
                AND b."Booking Date" < '2024-04-20'
              ORDER BY b."Booking Date", b."Time Slot (HH:MM)"
              
            - For queries based on a particular month:
              SELECT b."Booking ID", b."Customer Name", s."Service Name",
                     b."Booking Date", b."Time Slot (HH:MM)", b."Price (INR)"
              FROM public.bookings b
              JOIN public.services s ON b."Service ID" = s."Service ID"
              WHERE b."Customer Name" = 'user_id'
                AND TO_CHAR(b."Booking Date", 'YYYY-MM') = '2024-04'
              ORDER BY b."Booking Date", b."Time Slot (HH:MM)"
              
            Your response should include:
            1. The executed SQL query.
            2. A summary of the bookings retrieved:
               - Listing each booking with Service Name, Booking Date, Time Slot (HH:MM), and Price (INR).
            3. If the query is for future bookings, include a reminder about the cancellation policy.
            4. If the query is for previous bookings, ask if the user would like to rebook the same service.
        """)

        # Create the agent with a key rotation mechanism
        try:
            booking_query_agent = get_agno_agent_with_retry(
                instructions=booking_query_instructions,
                tools=[sql_tools]
            )
        except Exception as e:
            logger.error(f"Failed to create booking query agent after retries: {str(e)}")
            return ("I'm sorry, but I'm having trouble accessing the booking system right now. "
                    "Please try again later or contact us directly at (555) 123-4567 for booking information.")

        # Build the initial prompt, mentioning the query type and ensuring user_id is used as Customer Name.
        booking_query_prompt = f"User wants to query their {query_type} bookings. "
        booking_query_prompt += (f"The user id is '{customer_name}' which must be used as the exact value for the "
                                 f"'Customer Name' column in the SQL query. ")

        # If a date filter was detected, add instructions to apply any type of date filter based on the user request.
        if date_filter:
            formatted_date = format_date_for_postgresql(date_filter['date'])
            date_filter['formatted_date'] = formatted_date
            
            # Instruct the agent to analyze the relation (e.g., "before", "after", "month", etc.).
            booking_query_prompt += (
                f"The user's request includes a date filter: bookings {date_filter['relation']} "
                f"{date_filter['date']} (formatted as {formatted_date}). "
                "Please apply the appropriate SQL condition to filter the 'Booking Date'. "
            )
        else:
            booking_query_prompt += (
                "There is no explicit date filter in the request. Return all relevant bookings as per the query type. "
            )
            
        # Adjust further instructions based on query_type
        if query_type == "previous":
            booking_query_prompt += (
                "Query for previous bookings (booking dates before CURRENT_DATE). Summarize and ask if they want to rebook. "
            )
        elif query_type == "future":
            booking_query_prompt += (
                "Query for upcoming bookings (booking dates on or after CURRENT_DATE). Summarize and remind about cancellation policy. "
            )
        else:
            booking_query_prompt += (
                "Query for all bookings (both previous and upcoming). Distinguish between past and future appointments when summarizing. "
            )

        booking_query_prompt += "\n\nIMPORTANT: Always use the 'YYYY-MM-DD' format for dates in your SQL queries."

        # Attempt to get a response from the Agno agent.
        max_retries = min(3, len(gemini_api_keys) * 2)
        for retry in range(max_retries):
            try:
                if retry > 0:
                    logger.info(f"Retrying booking query agent with new API key/model (attempt {retry+1})")
                    rotate_gemini_key_and_model()
                    booking_query_agent = get_agno_agent(instructions=booking_query_instructions, tools=[sql_tools])
                
                response = booking_query_agent.run(booking_query_prompt)
                agent_response = response.content
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")
                
                logger.info(f"Booking query interaction - User: {user_id}, Query type: {query_type}, "
                            f"Response: {agent_response[:100]}...")
                return agent_response
            except Exception as inner_e:
                logger.error(f"Error from Agno agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    continue
                else:
                    return ("I'm sorry, but I'm having trouble retrieving your bookings at the moment. "
                            "Please try again later or contact us directly at (555) 123-4567 for booking information.")

    except Exception as e:
        logger.error(f"Error handling booking query with Agno: {str(e)}")
        return ("I'm sorry, but I'm having trouble accessing the booking system right now. "
                "Please try again later or contact us directly at (555) 123-4567 for booking information.")

# Extract date filter from message
def extract_date_filter(message):
    """
    Extract date filter from message (e.g., "after April 20", "before May 15", "in April 2024")
    
    Args:
        message (str): User message
        
    Returns:
        dict or None: Dictionary with date filter information or None if no filter found
    """
    message_lower = message.lower()
    
    # Define patterns for date filters with more flexible date formats
    before_patterns = [
        # Month name followed by day and optional year
        r'before\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # before April 20, 2024 or before April 20 2024
        r'prior\s+to\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # prior to April 20
        r'earlier\s+than\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # earlier than April 20
        r'up\s+to\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # up to April 20
        r'until\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # until April 20
        
        # ISO format dates
        r'before\s+(\d{4}-\d{2}-\d{2})',  # before 2024-04-20
        r'prior\s+to\s+(\d{4}-\d{2}-\d{2})',  # prior to 2024-04-20
        r'earlier\s+than\s+(\d{4}-\d{2}-\d{2})',  # earlier than 2024-04-20
        r'up\s+to\s+(\d{4}-\d{2}-\d{2})',  # up to 2024-04-20
        r'until\s+(\d{4}-\d{2}-\d{2})',  # until 2024-04-20
        
        # MM/DD/YYYY format
        r'before\s+(\d{1,2}/\d{1,2}/\d{4})',  # before 04/20/2024
        r'prior\s+to\s+(\d{1,2}/\d{1,2}/\d{4})',  # prior to 04/20/2024
        
        # Just the date without a relation word (assuming it's at the end of the message)
        r'(?:.*\s)(\w+\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})$',  # ... April 23 2024
        r'(?:.*\s)(\w+\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4})$',  # ... April 23, 2024
    ]
    
    after_patterns = [
        # Month name followed by day and optional year
        r'after\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # after April 20, 2024 or after April 20 2024
        r'later\s+than\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # later than April 20
        r'following\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # following April 20
        r'since\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # since April 20
        r'from\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # from April 20
        
        # ISO format dates
        r'after\s+(\d{4}-\d{2}-\d{2})',  # after 2024-04-20
        r'later\s+than\s+(\d{4}-\d{2}-\d{2})',  # later than 2024-04-20
        r'following\s+(\d{4}-\d{2}-\d{2})',  # following 2024-04-20
        r'since\s+(\d{4}-\d{2}-\d{2})',  # since 2024-04-20
        r'from\s+(\d{4}-\d{2}-\d{2})',  # from 2024-04-20
        
        # MM/DD/YYYY format
        r'after\s+(\d{1,2}/\d{1,2}/\d{4})',  # after 04/20/2024
        r'later\s+than\s+(\d{1,2}/\d{1,2}/\d{4})',  # later than 04/20/2024
    ]
    
    # On date patterns for queries like "on April 28" or "on 28 April"
    on_patterns = [
        # Month name followed by day and optional year
        r'on\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',  # on April 20, 2024 or on April 20 2024
        r'on\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+(?:,?\s+\d{4})?)',  # on 28 April, 2024 or on 28 April 2024
        
        # ISO format dates
        r'on\s+(\d{4}-\d{2}-\d{2})',  # on 2024-04-20
        
        # MM/DD/YYYY format
        r'on\s+(\d{1,2}/\d{1,2}/\d{4})',  # on 04/20/2024
        r'on\s+(\d{1,2}/\d{1,2})',  # on 04/20
        
        # Day and month only
        r'on\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+)',  # on 28 April
        r'on\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)',  # on April 28
    ]
    
    # Month patterns for queries like "in April 2024" or "for April"
    month_patterns = [
        r'in\s+(\w+)(?:\s+(\d{4}))?',  # in April, in April 2024
        r'for\s+(\w+)(?:\s+(\d{4}))?',  # for April, for April 2024
        r'during\s+(\w+)(?:\s+(\d{4}))?',  # during April, during April 2024
        r'of\s+(\w+)(?:\s+(\d{4}))?',  # of April, of April 2024
        r'(\w+)\s+(\d{4})',  # April 2024
        r'(\d{4})-(\d{2})'  # 2024-04
    ]
    
    # Check for month patterns first (most specific)
    for pattern in month_patterns:
        matches = re.search(pattern, message_lower)
        if matches:
            if len(matches.groups()) == 2 and matches.group(2):
                # Pattern with month and year
                month_str = matches.group(1)
                year_str = matches.group(2)
                
                # Handle numeric month format (2024-04)
                if re.match(r'\d{4}', month_str) and re.match(r'\d{2}', year_str):
                    year_str = month_str  # The first group is actually the year
                    month_str = year_str  # The second group is the month
                    month_year = f"{year_str}-{month_str}"
                else:
                    # Try to convert month name to number
                    try:
                        if len(month_str) <= 2 and month_str.isdigit():
                            # Already a numeric month
                            month_num = int(month_str)
                        else:
                            # Convert month name to date object to get month number
                            month_date = datetime.strptime(month_str, '%B')
                            month_num = month_date.month
                        
                        # Format as YYYY-MM
                        month_year = f"{year_str}-{month_num:02d}"
                    except Exception as e:
                        logger.warning(f"Error parsing month '{month_str}': {str(e)}")
                        # Use original strings if parsing fails
                        month_year = f"{year_str}-{month_str}"
            else:
                # Only month provided, use current year
                month_str = matches.group(1)
                try:
                    if len(month_str) <= 2 and month_str.isdigit():
                        # Already a numeric month
                        month_num = int(month_str)
                    else:
                        # Convert month name to date object to get month number
                        month_date = datetime.strptime(month_str, '%B')
                        month_num = month_date.month
                    
                    # Use 2025 as the default year for bookings
                    month_year = f"2025-{month_num:02d}"
                except Exception as e:
                    logger.warning(f"Error parsing month '{month_str}': {str(e)}")
                    # Use original string with default year if parsing fails
                    month_year = f"2025-{month_str}"
            
            return {
                'relation': 'month',
                'date': month_year
            }
    
    # Check for "before" patterns
    for pattern in before_patterns:
        matches = re.search(pattern, message_lower)
        if matches:
            date_str = matches.group(1)
            # Use the format_date_for_postgresql function to standardize the date
            formatted_date = format_date_for_postgresql(date_str)
            if formatted_date and re.match(r'\d{4}-\d{2}-\d{2}', formatted_date):
                date_str = formatted_date
            else:
                # Fallback to direct parsing if the function fails
                try:
                    # Remove ordinal suffixes
                    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                    
                    # Try different date formats
                    if ',' in date_str and re.search(r'\d{4}', date_str):  # Format: April 23, 2024
                        try:
                            date_obj = datetime.strptime(date_str, '%B %d, %Y')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(date_str, '%b %d, %Y')
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                pass
                    elif re.search(r'\d{4}', date_str):  # Format: April 23 2024
                        try:
                            date_obj = datetime.strptime(date_str, '%B %d %Y')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(date_str, '%b %d %Y')
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                pass
                    else:  # No year, use 2025 for bookings
                        try:
                            date_obj = datetime.strptime(f"{date_str} 2025", '%B %d %Y')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(f"{date_str} 2025", '%b %d %Y')
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                pass
                except Exception as e:
                    logger.warning(f"Error parsing date '{date_str}': {str(e)}")
                    # Keep the original string if parsing fails
                    pass
            
            return {
                'relation': 'before',
                'date': date_str
            }
    
    # Check for "after" patterns
    for pattern in after_patterns:
        matches = re.search(pattern, message_lower)
        if matches:
            date_str = matches.group(1)
            # Use the format_date_for_postgresql function to standardize the date
            formatted_date = format_date_for_postgresql(date_str)
            if formatted_date and re.match(r'\d{4}-\d{2}-\d{2}', formatted_date):
                date_str = formatted_date
            else:
                # Fallback to direct parsing if the function fails
                try:
                    # Remove ordinal suffixes
                    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                    
                    # Try different date formats
                    if ',' in date_str and re.search(r'\d{4}', date_str):  # Format: April 23, 2024
                        try:
                            date_obj = datetime.strptime(date_str, '%B %d, %Y')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(date_str, '%b %d, %Y')
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                pass
                    elif re.search(r'\d{4}', date_str):  # Format: April 23 2024
                        try:
                            date_obj = datetime.strptime(date_str, '%B %d %Y')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(date_str, '%b %d %Y')
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                pass
                    else:  # No year, use 2025 for bookings
                        try:
                            date_obj = datetime.strptime(f"{date_str} 2025", '%B %d %Y')
                            date_str = date_obj.strftime('%Y-%m-%d')
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(f"{date_str} 2025", '%b %d %Y')
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except ValueError:
                                pass
                except Exception as e:
                    logger.warning(f"Error parsing date '{date_str}': {str(e)}")
                    # Keep the original string if parsing fails
                    pass
            
            return {
                'relation': 'after',
                'date': date_str
            }
    
    return None

# Analyze message for intents and entities
def analyze_message(message):
    """
    Analyze a message for intents and entities

    Args:
        message (str): User message

    Returns:
        dict: Dictionary with intents and entities
    """
    # Initialize results
    results = {
        "intents": [],
        "service_entities": [],
        "date_entities": [],
        "time_entities": [],
        "customer_entities": []
    }

    # Convert message to lowercase for easier matching
    message_lower = message.lower()

    # Check for booking-related intents
    if any(word in message_lower for word in ["book", "appointment", "schedule", "reserve"]):
        results["intents"].append("booking")

    # Check for booking query intents
    if any(phrase in message_lower for phrase in ["my bookings", "my appointments", "check bookings",
                                                 "view bookings", "show bookings", "see my bookings"]):
        results["intents"].append("booking_query")

        # Check for specific booking query types
        if any(word in message_lower for word in ["previous", "past", "earlier", "before", "history"]):
            results["intents"].append("previous_bookings")
        if any(word in message_lower for word in ["future", "upcoming", "next", "scheduled", "coming"]):
            results["intents"].append("future_bookings")

    # Check for service-related intents
    if any(word in message_lower for word in ["service", "treatment", "massage", "facial", "spa", "price", "cost", "how much"]):
        results["intents"].append("service_query")

    # Check for general information intents
    if any(phrase in message_lower for phrase in ["what is", "tell me about", "explain", "how does", "benefits of"]):
        results["intents"].append("information_query")

    # Extract service entities (simplified - in a real app, would use NER)
    # Get services from database
    services = get_services_from_db()
    for service in services:
        service_name = service.get("name", "").lower()
        if service_name and service_name in message_lower:
            results["service_entities"].append(service.get("name"))

    # Extract date entities (simplified)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?\b',  # Month Day
        r'\b(?:tomorrow|today|next week|next month)\b'  # Relative dates
    ]

    date_entities = []
    for pattern in date_patterns:
        matches = re.findall(pattern, message_lower)
        for match in matches:
            # Format the date for PostgreSQL
            formatted_date = format_date_for_postgresql(match)
            date_entities.append(formatted_date)
    
    # Add the formatted dates to results
    results["date_entities"] = date_entities

    # Extract time entities (simplified)
    time_patterns = [
        r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b',  # HH:MM am/pm
        r'\b\d{1,2}\s*(?:am|pm)\b',         # HH am/pm
        r'\b(?:morning|afternoon|evening)\b'  # Time of day
    ]

    for pattern in time_patterns:
        matches = re.findall(pattern, message_lower)
        results["time_entities"].extend(matches)

    # Extract customer names (simplified)
    # Look for patterns like "for John Doe" or "customer John Doe"
    customer_patterns = [
        r'for\s+([A-Za-z\s]+?)(?:\s+on|\s+at|\s+with|\.|,|\s+$)',
        r'customer\s+([A-Za-z\s]+?)(?:\s+on|\s+at|\s+with|\.|,|\s+$)',
        r'client\s+([A-Za-z\s]+?)(?:\s+on|\s+at|\s+with|\.|,|\s+$)'
    ]

    for pattern in customer_patterns:
        matches = re.findall(pattern, message)
        for match in matches:
            if match.strip() and len(match.strip()) > 2:  # Avoid single letters
                results["customer_entities"].append(match.strip())

    return results

# Process user message
def process_message(user_id, message):
    """
    Process a user message and generate a response

    Args:
        user_id (str): User ID
        message (str): User message

    Returns:
        str: Response message
    """
    try:
        # Analyze the message for intents and entities
        analysis = analyze_message(message)
        intents = analysis.get("intents", [])

        # Check if this is a booking-related intent
        if "booking" in intents or "booking_query" in intents or any(word in message.lower() for word in ["book", "appointment", "schedule", "reserve", "my bookings"]):
            # Use Gemini with Agno to classify the booking message type
            booking_type = classify_booking_message(message)
            logger.info(f"Booking message classified as: {booking_type}")
            
            # Extract entities
            service_entities = analysis.get("service_entities", [])
            date_entities = analysis.get("date_entities", [])
            time_entities = analysis.get("time_entities", [])
            
            if booking_type == "booking_retrieval":
                # Determine the type of booking query
                if "previous_bookings" in intents or any(word in message.lower() for word in ["previous", "past", "earlier", "history"]):
                    query_type = "previous"
                # Check for words indicating future/upcoming bookings
                elif "future_bookings" in intents or any(word in message.lower() for word in ["future", "upcoming", "next", "scheduled", "coming"]):
                    query_type = "future"
                # If "before" is used with a date but without past-tense indicators, assume future bookings
                elif "before" in message.lower() and not any(word in message.lower() for word in ["previous", "past", "earlier", "history"]):
                    query_type = "future"
                # If "after" is used with "previous" or past-tense indicators, ensure it's treated as previous bookings
                elif "after" in message.lower() and any(word in message.lower() for word in ["previous", "past", "earlier", "history"]):
                    query_type = "previous"
                else:
                    query_type = "all"
                
                # Use the user_id as the customer name to filter bookings
                # This ensures users can only see their own bookings
                customer_name = user_id
                
                # Handle the booking query with Agno
                return handle_booking_query_with_agno(user_id, message, query_type, customer_name)
            else:  # booking_type == "create_booking"
                # Handle the booking creation with Agno
                return handle_booking_with_agno(user_id, message, intents, service_entities, date_entities, time_entities)
        
        # For other intents, use Agno with SQLTools and web search
        else:
            return handle_general_query_with_agno(user_id, message)

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return "I'm sorry, but I encountered an error processing your request. Please try again or contact our staff for assistance."


# Chat interface with improved context handling
def chat():
    user_id = input("Enter your user ID: ").strip()
    print(f"Tranquility Spa Assistant: Welcome to Tranquility Spa & Wellness Center, {user_id}! How can I help you today?")

    while True:
        try:
            message = input("You: ")
        except KeyboardInterrupt:
            print("\nTransquility Spa Assistant: Thank you for chatting with us. Have a relaxing day!")
            break

        if message.lower() in ["exit", "quit", "bye"]:
            print("Tranquility Spa Assistant: Thank you for chatting with us. Have a relaxing day!")
            break

        response = process_message(user_id, message)
        print("Tranquility Spa Assistant:", response)

# Handle general queries with Agno
def handle_general_query_with_agno(user_id, message):
    """
    Handle general queries using Agno agent with SQLTools and web search

    Args:
        user_id (str): User ID
        message (str): User message

    Returns:
        str: Response message from Agno agent
    """
    try:
        # Create database connection string
        db_url = f"postgresql+psycopg://{os.getenv('user')}:{os.getenv('password')}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('dbname')}"

        # Create SQL tools for Agno
        sql_tools = SQLTools(db_url=db_url)

        # Get the previous message from conversation history
        previous_message = ""
        conversation_history = get_conversation_history(user_id, limit=1)
        if conversation_history:
            previous_message = conversation_history[0].get('metadata', {}).get('message', '')
            logger.info(f"Previous message found: {previous_message[:50]}...")

        # Create web search tool if Serper API key is available
        tools = [sql_tools]

        # Create general query agent with specific instructions
        general_query_instructions = dedent("""
            You are an assistant for Tranquility Spa & Wellness Center.
            Your task is to help users with their questions about spa services and related topics.

            IMPORTANT INSTRUCTIONS:
            1. ALWAYS check the database FIRST for information about spa services, prices, etc.
            2. Only use web search if the question cannot be answered from the database
            3. For service-related questions, query the services table
            4. For booking-related questions, direct the user to make a booking
            5. Only search the web for topics related to spa services, wellness, or beauty treatments
            6. When searching the web, consider both the current and previous user messages for context - current message is question if question seems incomplete like maybe it has words like this or it use previous for context about what user is referring to
            7. When searching the web, dont mention searching in web and provide generic answer if not found
            
            DATABASE SCHEMA (EXACT NAMES - DO NOT MODIFY):
            - Table name: public.services
              Columns: "Service ID", "Service Name", "Description", "Price (INR)", "Category"
              Note: Column names include spaces and must be quoted in SQL queries

            - Table name: public.bookings
              Columns: "Booking ID", "Customer Name", "Service ID", "Booking Date", "Time Slot (HH:MM)", "Price (INR)"
              Note: The "Customer Name" column contains the user ID of the customer

            EXAMPLE SQL QUERIES:
            - To find information about a specific service:
              ```sql
              SELECT * FROM public.services WHERE "Service Name" ILIKE '%facial%'
              ```

            - To get all services in a category:
              ```sql
              SELECT * FROM public.services WHERE "Category" ILIKE '%massage%'
              ```

            - To get all available services:
              ```sql
              SELECT * FROM public.services ORDER BY "Category", "Service Name"
              ```

            RESPONSE GUIDELINES:
            1. For service questions, provide details about the service, price, and description
            2. For general spa questions, provide helpful information based on the database or web search
            3. Always be professional, friendly, and concise
            4. If suggesting a service, mention how the user can book it

            IMPORTANT: Only search the web if the question is related to spa services, wellness, or beauty treatments AND cannot be answered from the database.
        """)

        # Add Serper web search capability if API key is available
        if serper_api_key:
            # Create a custom function to search with Serper API
            def search_with_serper(query):
                """
                Search the web using Serper API

                Args:
                    query (str): Search query

                Returns:
                    dict: Search results
                """
                try:
                    headers = {
                        'X-API-KEY': serper_api_key,
                        'Content-Type': 'application/json'
                    }
                    payload = json.dumps({
                        "q": query,
                        "num": 5  # Get top 5 results
                    })
                    response = requests.post('https://google.serper.dev/search', headers=headers, data=payload)
                    return response.json()
                except Exception as e:
                    logger.error(f"Error searching with Serper: {str(e)}")
                    return {"error": str(e)}

            # Add the search function to the instructions
            general_query_instructions += dedent("""

                WEB SEARCH GUIDELINES:
                1. Only use web search if the database doesn't have the information
                2. Focus searches on spa, wellness, and beauty-related topics
                3. Prioritize reputable sources like wellness websites, medical sources, or spa industry publications
                4. Clearly indicate when information comes from web search vs. the database
                5. Do not search for unrelated topics or sensitive information
                6. When search results are provided, you MUST use them to answer the question
                7. Never claim you cannot access search results or external websites

                To search the web, use this format:
                ```
                I need to search for: [your search query]
                ```

                I will then perform the search and provide the results, which you MUST use to answer the user's question.
            """)
        else:
            search_with_serper = None

        # Create the Agno agent with key and model rotation on failure
        try:
            general_query_agent = get_agno_agent_with_retry(
                instructions=general_query_instructions,
                tools=tools
            )

            # If we have a Serper API key, we need to handle the search manually
            # since we're not using a dedicated Agno tool for web search
            if serper_api_key:
                logger.info("Serper API key available - will handle web searches manually")
        except Exception as e:
            logger.error(f"Failed to create general query agent after retries: {str(e)}")
            return "I'm sorry, but I'm having trouble accessing our information system right now. Please try again later or contact us directly at (555) 123-4567 for assistance."

        # Prepare the prompt for the general query agent
        general_query_prompt = f"User has asked: '{message}'. "
        
        # Include previous message for context if available
        if previous_message:
            general_query_prompt += f"Their previous message was: '{previous_message}'. "
        
        general_query_prompt += "First check our database for relevant information about our services. "

        if serper_api_key:
            general_query_prompt += "If the database doesn't have the answer and the question is related to spa services, wellness, or beauty treatments, indicate that you need to search the web by saying 'I need to search for: [your search query]'. "
            general_query_prompt += "When I provide web search results, you MUST use them to answer the question. Never claim you cannot access search results or external websites. "

        general_query_prompt += "Provide a helpful, accurate response based on all available information."

        # Get response from Agno agent with retry
        max_retries = min(3, len(gemini_api_keys) * 2)
        for retry in range(max_retries):
            try:
                # If this is a retry, get a new agent with a fresh API key and model
                if retry > 0:
                    logger.info(f"Retrying general query agent with new API key/model (attempt {retry+1})")
                    rotate_gemini_key_and_model()
                    general_query_agent = get_agno_agent(instructions=general_query_instructions, tools=tools)

                # Try to get a response
                response = general_query_agent.run(general_query_prompt)
                agent_response = response.content

                # Check if the agent wants to search the web
                if serper_api_key and "I need to search for:" in agent_response:
                    # Extract the search query
                    search_query_match = re.search(r'I need to search for:\s*(.+?)(?:\n|$)', agent_response)
                    if search_query_match:
                        search_query = search_query_match.group(1).strip()
                        
                        # Enhance search query with previous message context if available
                        enhanced_query = search_query
                        if previous_message:
                            # Combine current and previous messages for context-aware search
                            # Extract key terms from both messages to create a more focused query
                            combined_context = f"{previous_message} {message}"
                            # Use the original search query but enhance it with context
                            enhanced_query = f"{search_query} {combined_context}"
                            # Limit query length to avoid issues with search API
                            if len(enhanced_query) > 300:
                                enhanced_query = enhanced_query[:300]
                            
                            logger.info(f"Enhanced search query with previous message context")
                        
                        logger.info(f"Performing web search for: {enhanced_query}")

                        # Perform the search with the enhanced query
                        search_results = search_with_serper(enhanced_query)

                        # Format the search results
                        formatted_results = "### WEB SEARCH RESULTS ###\n\n"

                        if "error" in search_results:
                            formatted_results += f"Error performing search: {search_results['error']}"
                        else:
                            # Add organic results
                            if "organic" in search_results:
                                formatted_results += "ORGANIC SEARCH RESULTS:\n"
                                for i, result in enumerate(search_results["organic"][:3], 1):
                                    title = result.get("title", "No title")
                                    link = result.get("link", "No link")
                                    snippet = result.get("snippet", "No description")
                                    formatted_results += f"RESULT {i}:\nTitle: {title}\nContent: {snippet}\nSource: {link}\n\n"

                            # Add knowledge graph if available
                            if "knowledgeGraph" in search_results:
                                kg = search_results["knowledgeGraph"]
                                if "title" in kg:
                                    formatted_results += "KNOWLEDGE GRAPH INFORMATION:\n"
                                    formatted_results += f"Title: {kg.get('title')}\n"
                                    if "description" in kg:
                                        formatted_results += f"Description: {kg.get('description')}\n\n"
                                        
                            formatted_results += "### END OF SEARCH RESULTS ###\n\n"

                        # Send the search results back to the agent for processing
                        context_info = ""
                        if previous_message:
                            context_info = f"The user's previous message was: '{previous_message}'. "
                        
                        follow_up_prompt = f"""
Based on the user's question: '{message}', {context_info}I performed a web search for '{enhanced_query}' and found the following information:

{formatted_results}

IMPORTANT: You MUST use the web search results above to answer the user's question. Do not claim you cannot access the search results or external websites.
If the search results contain relevant information, summarize it and cite the sources.
If the search results don't directly answer the question, say so and provide the best information you can based on what was found.

Please provide a helpful response focusing on spa and wellness-related information.
"""

                        # Get the final response with the search results
                        follow_up_response = general_query_agent.run(follow_up_prompt)
                        agent_response = follow_up_response.content

                # Make sure we have a valid response
                if agent_response is None:
                    raise ValueError("Agno agent returned None response")

                # Log the general query interaction
                logger.info(f"General query interaction - User: {user_id}, Query: {message[:50]}..., Response: {agent_response[:100]}...")

                # Store the interaction in memory
                store_memory(user_id, message, agent_response)

                return agent_response
            except Exception as inner_e:
                logger.error(f"Error getting response from Agno agent (attempt {retry+1}/{max_retries}): {str(inner_e)}")
                if retry < max_retries - 1:
                    # Will retry with a new key and model
                    continue
                else:
                    # All retries failed, return a fallback response
                    return "I'm sorry, but I'm having trouble processing your question right now. Please try asking in a different way or contact our staff directly at (555) 123-4567 for assistance."

    except Exception as e:
        logger.error(f"Error handling general query with Agno: {str(e)}")
        return "I'm sorry, but I encountered an error while trying to answer your question. Please try again or contact our staff for assistance."

# Entry point
if __name__ == "__main__":
    chat()
