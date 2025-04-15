import logging
import re
from datetime import datetime
from textwrap import dedent
from mem.emb import get_embedding, cosine_similarity
# Import specific functions from memory to avoid circular imports
from memory import get_conversation_history, get_adaptive_conversation_history
from agent_init import get_agno_agent_with_retry, get_agno_agent, rotate_gemini_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
