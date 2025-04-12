import os
import logging
from agno.agent import Agent
from agno.models.xai import xAI
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Logging and environment setup
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ----------------------------------------------------------------------
# GROK AGENT SETUP
# ----------------------------------------------------------------------
def get_grok_agent(instructions=None):
    """Create an Agno agent with Grok."""
    default_instructions = (
        "You are a helpful assistant for Tranquility Spa & Wellness Center. "
        "Provide accurate and friendly information about spa services, hours, and bookings. "
        "Keep responses concise and professional."
    )

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        logger.error("XAI_API_KEY not set. Please set it in your environment or .env file.")
        return None

    try:
        return Agent(
            model=xAI(id="grok-1", api_key=api_key),
            instructions=instructions or default_instructions,
            markdown=True
        )
    except Exception as e:
        logger.error(f"Failed to create Grok agent: {str(e)}")
        return None

# ----------------------------------------------------------------------
# SPA BUSINESS INFORMATION
# ----------------------------------------------------------------------
SPA_INFO = {
    "name": "Tranquility Spa & Wellness Center",
    "hours": {
        "Monday": "9:00 AM - 8:00 PM",
        "Tuesday": "9:00 AM - 8:00 PM",
        "Wednesday": "9:00 AM - 8:00 PM",
        "Thursday": "9:00 AM - 8:00 PM",
        "Friday": "9:00 AM - 9:00 PM",
        "Saturday": "8:00 AM - 9:00 PM",
        "Sunday": "10:00 AM - 6:00 PM"
    },
    "services": [
        {"name": "Swedish Massage", "price": 1200, "description": "A relaxing full-body massage."},
        {"name": "Glowing Facial", "price": 1500, "description": "A rejuvenating facial treatment."},
        {"name": "Hot Stone Therapy", "price": 1800, "description": "Warm stones for deep relaxation."}
    ],
    "booking_policy": "Bookings require 4 hours advance notice. Call (555) 123-4567 to reserve."
}

# ----------------------------------------------------------------------
# MESSAGE PROCESSING
# ----------------------------------------------------------------------
def process_message(user_id: str, message: str, agent: Agent) -> str:
    """Process user messages with Grok."""
    if not agent:
        return "Sorry, I'm having trouble connecting right now. Please try again later."

    # Build context with spa info
    context = (
        f"You are assisting {user_id} at {SPA_INFO['name']}.\n\n"
        "Spa Information:\n"
        f"Hours: {', '.join([f'{day}: {hours}' for day, hours in SPA_INFO['hours'].items()])}\n"
        "Services:\n" + "\n".join([f"- {svc['name']} (₹{svc['price']}): {svc['description']}" for svc in SPA_INFO['services']]) + "\n"
        f"Booking Policy: {SPA_INFO['booking_policy']}\n\n"
    )

    # Simple intent detection
    message_lower = message.lower()
    if "book" in message_lower or "appointment" in message_lower:
        prompt = f"{context}User: {message}\n\nHelp the user book a service. If details are missing, ask for them."
    elif "service" in message_lower or "massage" in message_lower or "facial" in message_lower:
        prompt = f"{context}User: {message}\n\nProvide details about our services."
    elif "hours" in message_lower or "open" in message_lower:
        prompt = f"{context}User: {message}\n\nShare our operating hours."
    else:
        prompt = f"{context}User: {message}\n\nRespond helpfully based on the spa info."

    try:
        response = agent.run(prompt)
        return response.content if response and response.content else "I didn’t quite understand that. How can I assist you?"
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return "Sorry, I ran into an issue. Please try again!"

# ----------------------------------------------------------------------
# SIMPLE CHAT LOOP
# ----------------------------------------------------------------------
def chat():
    user_id = input("Enter your user ID: ").strip()
    print(f"Welcome to {SPA_INFO['name']}, {user_id}! How can I assist you today?")

    agent = get_grok_agent()
    if not agent:
        print("Sorry, I can’t start the chat without a valid setup. Check your XAI_API_KEY.")
        return

    while True:
        try:
            message = input("You: ")
        except KeyboardInterrupt:
            print("\nThanks for visiting! Have a relaxing day!")
            break

        if message.lower() in ["exit", "quit", "bye"]:
            print("Thanks for chatting! Enjoy your day!")
            break

        response = process_message(user_id, message, agent)
        print(f"{SPA_INFO['name']} Assistant: {response}")

if __name__ == "__main__":
    chat()