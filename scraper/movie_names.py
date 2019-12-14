from bs4 import BeautifulSoup
import requests 
import string 

url = "https://www.traileraddict.com/thefilms/" 

def get_from_url(postfix):
    global url 
    movies = []
    new_url = url + postfix 
    html_doc = requests.get(new_url).text 
    page_soup = BeautifulSoup(html_doc, "html.parser")
    for class_type in ["film-abc-list first column", "film-abc-list second column"]: 
        lst_html =  "<html>" + page_soup.find("ul", class_=class_type).__str__() + "</html>" 
        lst_soup = BeautifulSoup(lst_html, "html.parser")
        movies += [li.string for li in lst_soup.findAll("li")]
    return movies


res = []
for c in (["!#"] + list(string.ascii_lowercase)): 
    res += get_from_url(c)
with open("../movie_list.txt", "w") as f: 
    f.write("\n".join(map(str, res)))