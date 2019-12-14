from bs4 import BeautifulSoup
import requests

url = 'https://www.traileraddict.com/the-a-list/trailer'

html_doc = requests.get(url).text
soup = BeautifulSoup(html_doc, 'html.parser')

print(soup)