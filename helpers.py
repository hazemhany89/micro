"""
helpers.py

Utility functions shared across modules.
"""

def clean_keyword(keyword):
    """
    Cleans and normalizes the input keyword.

    Args:
        keyword (str): Raw keyword input.

    Returns:
        str: Cleaned keyword.
    """
    return keyword.strip().lower()