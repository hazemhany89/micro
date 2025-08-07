"""
core/fetch_google_news.py

Provides a function to fetch Google News articles for a keyword
with filtering options for date range and source.
"""

from GoogleNews import GoogleNews
from datetime import datetime, timedelta

def fetch_google_news(keyword: str, limit: int = 20, start_date: str = None, end_date: str = None, source: str = None) -> list:
    """
    Fetches Google News articles for a keyword with optional filtering.
    
    Args:
        keyword (str): The keyword to search for.
        limit (int): Maximum number of articles to fetch (default: 20).
        start_date (str): Start date for filtering in format 'MM/DD/YYYY' (default: 30 days ago).
        end_date (str): End date for filtering in format 'MM/DD/YYYY' (default: today).
        source (str): Filter articles by specific news source (default: None).
    
    Returns:
        list[dict]: List of article info dicts (title, source, date, url).
    """
    print(f"Fetching Google News for keyword: '{keyword}' (limit: {limit})")
    print(f"Date range: {start_date} to {end_date}, Source filter: {source}")
    
    # Set default date range if not provided (last 30 days)
    if not start_date:
        thirty_days_ago = datetime.now() - timedelta(days=30)
        start_date = thirty_days_ago.strftime('%m/%d/%Y')
    
    if not end_date:
        end_date = datetime.now().strftime('%m/%d/%Y')
    
    articles = []
    try:
        # Initialize GoogleNews
        googlenews = GoogleNews(lang='en', period='')
        
        # Set date range
        googlenews.set_time_range(start_date, end_date)
        
        # Set source if provided
        if source:
            googlenews.set_encode_uri(True)  # Ensure proper URL encoding
            googlenews.set_news_source(source)
        
        # Search for keyword
        googlenews.search(keyword)
        
        # Get results (first page)
        results = googlenews.results()
        
        # If we need more results and haven't reached the limit yet
        page = 2
        while len(results) < limit and page <= 3:  # Max 3 pages to avoid excessive requests
            googlenews.getpage(page)
            results.extend(googlenews.results())
            page += 1
        
        # Process and format results
        for article in results[:limit]:  # Limit to requested number
            articles.append({
                'title': article.get('title', 'No title'),
                'source': article.get('media', 'Unknown source'),
                'date': article.get('date', 'Unknown date'),
                'url': article.get('link', '#')
            })
        
        print(f"Successfully fetched {len(articles)} Google News articles for '{keyword}'.")
        return articles
    
    except Exception as e:
        print(f"An error occurred while fetching Google News for '{keyword}': {e}")
        return []
    
    finally:
        # Clear GoogleNews to avoid memory issues
        if 'googlenews' in locals():
            googlenews.clear()