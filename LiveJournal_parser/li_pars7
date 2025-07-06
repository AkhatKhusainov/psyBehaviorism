from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service
import time
from selenium.webdriver.firefox.options import Options


def main_li_pars7(name):
    options = Options()
    options.headless = False
    options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe" #путь до клиента браузера
    author = name 
    driver_path = 'D:\\downloads\\geckodriver-v0.35.0-win64\\geckodriver.exe' #путь до geckodriver
    service = Service(executable_path=driver_path)
    browser = webdriver.Firefox(service=service,options=options)

    profile_url = f"https://{author}.livejournal.com/calendar/"
    browser.get(profile_url)
    #browser.minimize_window()
    time.sleep(5)

    checkbox = browser.find_element(By.CSS_SELECTOR, 'input[type="checkbox"]#view-own')
    if not checkbox.is_selected():
        checkbox.click()

    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.j-l-wrapper')))

    years_elements = browser.find_elements(By.CSS_SELECTOR, 'li.j-years-nav-item')
    years = [year.text for year in years_elements]
    print(f"years:{years}")

    year_links = [f"https://{author}.livejournal.com/{year}/" for year in years]

    all_post_links = []
    i=0
    for year_link in year_links:
        print(years[i])
        i=+1
        browser.get(year_link)
        time.sleep(5.5)  

        months_elements = browser.find_elements(By.CSS_SELECTOR, 'div.j-calendar-month > a')
        #print(months_elements)
        month_links = [month.get_attribute("href") for month in months_elements]
        print(f"количество ссылок в месяце: {len(month_links)}")
        for month_link in month_links:
            browser.get(month_link)
            time.sleep(4.5)  
            post_elements = browser.find_elements(By.CSS_SELECTOR, 'li.j-day-subjects-item > a')
            post_links = [post.get_attribute("href") for post in post_elements]


            all_post_links.extend(post_links)
            print(len(all_post_links))

    with open(f'post_links_{author}.txt', 'w') as f:
        for link in all_post_links:
            f.write(link + "\n")

    browser.quit()

