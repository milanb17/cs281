import re
from bs4 import BeautifulSoup
import requests

def scrape(movie):
    for x in range(1,20):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        page = requests.get("https://www.rottentomatoes.com/m/"+movie, headers=headers)

        soup = BeautifulSoup(page.content, 'html.parser')

        score = soup.find("span", class_ = "mop-ratings-wrap__percentage").text
    return re.findall(r'\d+', score)[0]