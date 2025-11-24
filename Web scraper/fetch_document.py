"""
A module for downloading PDF documents from web URLs.

This module provides functionality to fetch and save PDF documents from specified URLs.
It includes basic HTTP request handling and file writing operations.

Functions:
    None (script is designed to run directly)

Usage:
    python fetch_document.py

Dependencies:
    - requests: For making HTTP requests
    - bs4 (BeautifulSoup): For potential HTML parsing (currently unused)

Example:
    The script downloads a PDF from SCB (Statistics Sweden) and saves it locally
    as 'report-2025.pdf'.

Notes:
    - The script currently downloads a single document from a hardcoded URL
    - Commented code includes examples for:
        * Handling relative URLs with base URL joining
        * Downloading multiple documents from a webpage
        * Parsing document links from HTML using BeautifulSoup
    - Error handling is implemented with HTTP status code checking
    - Downloaded files are saved in binary write mode ('wb')

Status Codes:
    - 200: Successful download
    - Other: Failed request (error message displayed)
"""
import requests
# from bs4 import BeautifulSoup
# import os

# Step 1: Send an HTTP request to the webpage
URL = "https://www.scb.se/contentassets/e990c2fbd1e14545804150efd6208bb9/le0108_kd_2025_v1_20251017.pdf"  # Replace with the actual URL
# response = requests.get(URL)
# Handle relative URLs
# base_url = 'https://example.com'  # The base URL of the website
# full_url = os.path.join(base_url, document_link)
# print('Full URL:', full_url)

# Step 3: Download the document
document_response = requests.get(URL, timeout=30)

# Check if the document request was successful
if document_response.status_code == 200:
    # Save the document to a file
    with open("Web scraper/report-2025.pdf", "wb") as file:
        file.write(document_response.content)
    print("Document downloaded successfully.")
else:
    print(
        "Failed to download the document. Status code:", document_response.status_code
    )

# Example: Downloading multiple documents
# Find all document links on the page
# document_links = soup.find_all('a', {'class': 'download-link'})

# Loop through each link and download the corresponding document
# for i, link in enumerate(document_links):
#    document_url = os.path.join(base_url, link['href'])
#    document_response = requests.get(document_url)
#
#    if document_response.status_code == 200:
#        # Save each document with a unique name
#        file_name = f'report_{i+1}.pdf'
#        with open(file_name, 'wb') as file:
#            file.write(document_response.content)
#        print(f'Document {i+1} downloaded successfully as {file_name}.')
#    else:
#        print(f'Failed to download document {i+1}. Status code:', document_response.status_code)