import re
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def extract_date_filter(message):
    """
    Extract date filter from message (e.g., "after April 20", "before May 15", "in April 2024")
    
    Args:
        message (str): User message
        
    Returns:
        dict or None: Dictionary with date filter information or None if no filter found
    """
    message_lower = message.lower()
    
    # Check for year-only patterns first (most specific for this fix)
    year_patterns = [
        r'for\s+(?:the\s+)?(?:year\s+)?(\d{4})',  # for the year 2025, for 2025
        r'in\s+(?:the\s+)?(?:year\s+)?(\d{4})',   # in the year 2025, in 2025
        r'during\s+(?:the\s+)?(?:year\s+)?(\d{4})',  # during the year 2025, during 2025
        r'(?:the\s+)?year\s+(\d{4})',  # the year 2025, year 2025
        r'(?<!\d)(\d{4})(?!\d)'  # standalone year like 2025 (not part of another number)
    ]
    
    # Special case for "for the year 2025" type queries
    if "for" in message_lower and re.search(r'\b\d{4}\b', message_lower):
        year_match = re.search(r'\b(\d{4})\b', message_lower)
        if year_match:
            year_str = year_match.group(1)
            logger.info(f"Detected year in 'for' query: {year_str}")
            return {
                'relation': 'year',
                'date': year_str
            }
    
    # Check for year patterns
    for pattern in year_patterns:
        matches = re.search(pattern, message_lower)
        if matches:
            year_str = matches.group(1)
            # Log the year detection for debugging
            logger.info(f"Detected year filter: {year_str}")
            return {
                'relation': 'year',
                'date': year_str
            }
    
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


def extract_entities_from_text(text, keywords):
    """
    Extract entities from text based on keywords
    
    Args:
        text (str): Text to extract entities from
        keywords (list): List of keywords to look for
        
    Returns:
        set: Set of extracted entities
    """
    entities = set()
    
    # Look for service names in the text
    for keyword in keywords:
        if keyword in text:
            # Try to extract the full service name
            # Look for patterns like "X facial", "Y massage", etc.
            patterns = [
                rf'([a-z\s]+{keyword})',  # e.g., "glowing facial"
                rf'({keyword}[a-z\s]+)',  # e.g., "facial treatment"
                rf'"([^"]*{keyword}[^"]*)"'  # Anything in quotes containing the keyword
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Clean up the match
                    clean_match = match.strip()
                    if 3 <= len(clean_match) <= 30:  # Reasonable length for a service name
                        entities.add(clean_match)
    
    return entities


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


