from selenium import webdriver
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

with open("movie_urls.txt") as f:
    with open("trailers.txt", 'w') as w:
        for movie in f:
            movie = movie.rstrip("\n").split("\t")[1]
            print(f"Getting {movie}....", end="")
            url = f"https://www.traileraddict.com/{movie}/trailer"
            driver.get(url)
            try:
                video_source = driver.find_element_by_id("trailerplayer_html5_api").get_attribute("src")
                w.write(f"{movie[1:]},{video_source}")
                print("success.")
            except:
                print(f"FAILED.")

driver.quit()