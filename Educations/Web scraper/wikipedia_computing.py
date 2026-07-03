"""
Wikipedia Cloud Computing Comparison Web Scraper

This script scrapes cloud computing comparison data from Wikipedia and saves it to a CSV file.
The script performs the following operations:

1. Sends an HTTP GET request to the Wikipedia cloud computing comparison page
2. Parses the HTML content using BeautifulSoup
3. Extracts the first table found on the page
4. Processes table headers and data rows
5. Creates a pandas DataFrame with the extracted data
6. Saves the data to a CSV file named 'scraped_data.csv'

Dependencies:
    - requests: For making HTTP requests
    - beautifulsoup4: For HTML parsing
    - pandas: For data manipulation and CSV export

Output:
    - scraped_data.csv: CSV file containing the extracted table data

Error Handling:
    - Checks HTTP response status code before proceeding
    - Prints success/failure messages for request status

Note:
    This script targets the first table found on the Wikipedia page.
    If the page structure changes, the script may need modification.
"""
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send an HTTP request to the webpage
URL = "https://en.wikipedia.org/wiki/Cloud-computing_comparison"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

try:
    response = requests.get(URL, headers=headers, timeout=10)
    response.raise_for_status()  # Raises an HTTPError for bad responses
except requests.exceptions.RequestException as e:
    print(f"Error occurred while making request: {e}")
    sys.exit(1)
print("Request successful with status code:", response.status_code)

# Parse the HTML content
try:
    soup = BeautifulSoup(response.content, "html.parser")
except (ValueError, TypeError) as e:
    print(f"Error occurred while parsing HTML: {e}")
    sys.exit(1)

# Print the title of the webpage to verify
print("Title: " + soup.title.text) # type: ignore

# Find the table containing the data (selecting the first table by default)
table = soup.find("table")

# Extract table rows
rows = table.find_all("tr") # type: ignore

# Extract headers from the first row (using <th> tags)
table_headers = [header.text.strip() for header in rows[0].find_all("th")]

# Loop through the rows and extract data (skip the first row with headers)
data = []
for row in rows[1:]:  # Start from the second row onwards
    cols = row.find_all("td")
    cols = [col.text.strip() for col in cols]
    data.append(cols)

# Convert the data into a pandas DataFrame, using the extracted headers as column names
df = pd.DataFrame(data, columns=table_headers)

# Display the first few rows of the DataFrame to verify
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv("Web scraper/scraped_data.csv", index=False)