import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://docs.confluent.io/platform/current/installation/system-requirements.html'  # Replace with the URL of the website you want to scrape
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print('Request successful!')
else:
    print('Failed to retrieve the webpage')

soup = BeautifulSoup(response.content, 'html.parser')

# Print the title of the webpage to verify
print(soup.title.text)

# Find the table containing the data
table = soup.find('table', {'id': 'docutils'})  # Replace 'data-table' with the actual id or class of the table
print(table)

# Extract table rows
#rows = table.find_all('tr')

# Loop through the rows and extract data
#data = []
#for row in rows:
#    cols = row.find_all('td')
#    cols = [col.text.strip() for col in cols]
#    data.append(cols)

# Convert the data into a pandas DataFrame for easier manipulation
#df = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])  # Replace with actual column names

# Display the scraped data
#print(df)

# Save the DataFrame to a CSV file
#df.to_csv('scraped_data.csv', index=False)