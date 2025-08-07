"""
core/fetch_google_trends.py

Provides a function to fetch the average demand score for a keyword
from Google Trends over a specified timeframe and country.
"""

from pytrends.request import TrendReq
import pandas as pd

def fetch_google_trends(keyword: str, geo: str = '', timeframe: str = 'today 3-m') -> float:
    """
    Fetches the average demand score for a keyword from Google Trends
    over a specified timeframe and geographical region.

    Args:
        keyword (str): The keyword to search for.
        geo (str): Geographical region (ISO code, e.g., 'US', 'EG', 'SA'). Use '' for Worldwide.
        timeframe (str): Timeframe for the trend data (e.g., 'today 1-m', 'today 3-m').

    Returns:
        float: The average demand score (0-100 scale), or 0 if no data is found or an error occurs.
    """
    print(f"Fetching Google Trends for keyword: '{keyword}', Geo: '{geo}', Timeframe: '{timeframe}'")
    try:
        # Increased timeout to 60 seconds for read, 10 for connect
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 60))
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
        data = pytrends.interest_over_time()

        if data.empty or keyword not in data.columns:
            print(f"No Google Trends data found for '{keyword}' with geo='{geo}' and timeframe='{timeframe}'. DataFrame is empty or keyword column missing.")
            return 0.0

        demand_score = float(data[keyword].mean())
        print(f"Successfully fetched demand score for '{keyword}': {demand_score}")
        return demand_score
    except AttributeError as e:
        print(f"AttributeError while fetching Google Trends for '{keyword}': {e}. This often means no data for the keyword or an issue with pytrends.")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred while fetching Google Trends for '{keyword}': {e}")
        return 0.0