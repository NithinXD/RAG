import uuid
import time
import json
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
from convo import process_message
from booking import handle_booking_with_agno, handle_booking_query_with_agno
from analysis import analyze_message, classify_booking_message
 
def chat():
    user_id = input("Enter your user ID: ").strip()
    print(f"Red Trends Spa Assistant: Welcome to Red Trends Spa & Wellness Center, {user_id}! How can I help you today?")

    while True:
        try:
            message = input("You: ")
        except KeyboardInterrupt:
            print("\nRed Trends Spa Assistant: Thank you for chatting with us. Have a relaxing day!")
            break

        if message.lower() in ["exit", "quit", "bye"]:
            print("Red Trends Spa Assistant: Thank you for chatting with us. Have a relaxing day!")
            break

        response = process_message(user_id, message)
        print("Red Trends Spa Assistant:", response)


# Entry point
if __name__ == "__main__":
    chat()