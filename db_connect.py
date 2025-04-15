
import os
import logging
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        cursor = connection.cursor()

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
