import os
import logging
from textwrap import dedent
from google import generativeai as genai
import agno
from agno.agent import Agent
from agno.models.google import Gemini
from config import get_gemini_model, GEMINI_MODELS, gemini_api_keys, current_key_index, current_model_index

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to rotate to the next key when one fails
def rotate_gemini_key():
    global current_key_index

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
        You are a helpful assistant for Red Trends Spa & Wellness Center.
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

            return agent
        except Exception as e:
            last_error = e
            logger.error(f"Error creating Agno agent (attempt {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                logger.info(f"Retrying with next Gemini API key or model")

    logger.error(f"All attempts to create Agno agent failed: {str(last_error)}")
    raise last_error