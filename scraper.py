from bs4 import BeautifulSoup
import requests
from os import system
import string
from joblib import Parallel, delayed
import re

def get_score(movie):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    page = requests.get("https://www.rottentomatoes.com/m/"+movie, headers=headers)

    soup = BeautifulSoup(page.content, 'html.parser')

    score = soup.find("span", class_ = "mop-ratings-wrap__percentage").text
    return re.findall(r'\d+', score)[0]

def scrape_with_letter(c):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        page = requests.get(f"http://www.hd-trailers.net/library/{c}/", headers = headers)

        soup = BeautifulSoup(page.content, 'html.parser')
        hrefs = [td.a['href'] for td in soup.find_all("td", class_ = "trailer")]
        for href in hrefs:
            try:
                print(f"Getting {href}...", end="")
                trailer_page = requests.get(f"http://www.hd-trailers.net{href}", headers = headers)
                trailer_soup = BeautifulSoup(trailer_page.content, 'html.parser')
                direct_download = trailer_soup.find("td", class_ = "bottomTableResolution").a['href']
                score = get_score(href.split('/')[-2])
                system(f"wget -c -o /dev/null -O trailers/{href.split('/')[-2]}--{score}.mp4 {direct_download}")
                print("done.")
            except:
                print("failed.")
    except:
        pass

Parallel(n_jobs = -1)(delayed(scrape_with_letter)(c) for c in '0' + string.ascii_lowercase)
