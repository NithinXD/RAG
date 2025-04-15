import os
import logging
from dotenv import load_dotenv
from google import generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Memory types
MEMORY_TYPES = {
    "INTERACTION": "interaction",           # Regular conversation
    "FAQ": "faq",                           # Frequently asked questions
    "BOOKING": "booking",                   # Booking information
    "PREFERENCE": "preference",             # User preferences
    "SERVICE_INTEREST": "service_interest"  # Services the user has shown interest in
}


# Spa service business information
SPA_BUSINESS = {
    "name": "Red Trends Spa & Wellness Center",
    "description": "A premium spa offering massage therapy, facials, body treatments, and wellness services.",
    "hours": {
        "Monday to Sumday": "9:00 AM - 7:00 PM"
    },
    "services": [],
    "booking_policy": "Appointments must be booked at least 4 hours in advance. We recommend booking 2-3 days ahead for weekend appointments.",
    "cancellation_policy": "Cancellations must be made at least 24 hours before your appointment to avoid a 50% cancellation fee. No-shows will be charged the full service amount.",
    "contact": {
        "phone": "+91 8838745128",
        "email": "appointments@Red Trendsspa.com",
        "website": "www.Red Trendsspa.com",
        "address": "1965 Relaxville, Madurai, Tamil Nadu 625017"
    }
}


# Get all available Gemini API keys
def get_gemini_api_keys():
    keys = []
    # Get the primary key
    primary_key = os.getenv("GEMINI_API_KEY")
    if primary_key:
        keys.append(primary_key)

    # Get additional keys that might be in the .env file
    for i in range(1, 10): 
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
                # check for keys in format GOOGLE_API_KEY=AIza...
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

    # over after trying all keys
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
    return genai.GenerativeModel(model_to_use)

# Initialize API keys
gemini_api_keys = get_gemini_api_keys()
serper_api_key = os.getenv("SERPER_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")

# Database configuration
DB_CONFIG = {
    "user": os.getenv("user"),
    "password": os.getenv("password"),
    "host": os.getenv("host"),
    "port": os.getenv("port"),
    "dbname": os.getenv("dbname")
}  
