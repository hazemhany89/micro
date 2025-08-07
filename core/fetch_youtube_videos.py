"""
core/fetch_youtube_videos.py

Provides a function to fetch the top YouTube videos for a keyword
using youtube-search-python, including publishing date and channel info.
"""

from youtubesearchpython import VideosSearch
import httpx

# Monkey patch httpx.post to handle the 'proxies' parameter issue
original_post = httpx.post

def patched_post(*args, **kwargs):
    # Remove 'proxies' parameter if present
    if 'proxies' in kwargs:
        del kwargs['proxies']
    return original_post(*args, **kwargs)

# Apply the patch
httpx.post = patched_post

def fetch_youtube_videos(keyword: str, limit: int = 30):
    """
    Fetches the top YouTube videos for a keyword, including titles, views, links,
    publishing dates, and channel names.

    Args:
        keyword (str): The keyword to search for.
        limit (int): Number of top videos to fetch (up to 50 for better analysis).

    Returns:
        list[dict]: List of video info dicts (title, views, link, published_time, channel_name).
    """
    print(f"Fetching YouTube videos for keyword: '{keyword}' (limit: {limit})")
    videos = []
    try:
        # youtube-search-python can fetch up to 100 results easily, but we'll respect the user's limit
        videos_search = VideosSearch(keyword, limit=limit)
        results = videos_search.result().get('result', [])
        print(f"Raw YouTube search results for '{keyword}' (first 2 results for brevity): {results[:2]}")

        for video in results:
            views = 0
            # Extracting view count and handling potential errors
            # It can be 'No views' or '1,234 views'
            view_count_text = video.get('viewCount', {}).get('text', '0 views').replace(',', '').replace(' views', '')
            try:
                # Attempt to convert to int, default to 0 if not a valid number
                views = int(view_count_text.strip().split(' ')[0]) # Take only the number part
            except ValueError:
                print(f"Could not parse view count for video '{video.get('title')}': '{view_count_text}'. Setting views to 0.")
                views = 0
            except Exception as e:
                print(f"Unexpected error parsing view count for video '{video.get('title')}': {e}. Setting views to 0.")
                views = 0

            videos.append({
                'title': video.get('title'),
                'views': views,
                'link': video.get('link'),
                'published_time': video.get('publishedTime'), # e.g., '1 month ago', '3 days ago'
                'channel_name': video.get('channel', {}).get('name')
            })
        print(f"Successfully parsed {len(videos)} YouTube videos for '{keyword}'.")
        return videos
    except Exception as e:
        print(f"An error occurred while fetching YouTube videos for '{keyword}': {e}")
        return []