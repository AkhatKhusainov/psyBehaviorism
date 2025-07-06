from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException, NoSuchElementException,
    StaleElementReferenceException, ElementNotInteractableException
)
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
import csv
import time
from selenium.webdriver.firefox.options import Options

options = Options()
options.headless = True

def get_livejournal_posts(author):
    driver_path = 'D:\\downloads\\geckodriver-v0.35.0-win64\\geckodriver.exe'
    service = Service(executable_path=driver_path, options=options)
    browser = webdriver.Firefox(service=service)

    url = f"https://pikabu.ru/{author}"
    browser.get(url)
    parsed_links = []
    parsed_content = []
    posts=[]

    def parse_data():
        print("parsing")
        post_elements = browser.find_elements(By.CSS_SELECTOR, 'h2.story__title a.story__title-link')
        post_containers = browser.find_elements(By.CLASS_NAME, 'story__content-inner')

        for post, container in zip(post_elements, post_containers):
            link = post.get_attribute('href')
            content = container.text.strip()  

            if link not in parsed_links and content not in parsed_content:
                parsed_links.append(link)
                parsed_content.append(content)
                posts.append((link, content))


    time.sleep(8)

    processed_buttons = set()  

    scroll_pause_time = 5
    scroll_step = 1000
    last_height = browser.execute_script("return document.body.scrollHeight")

    while True:
        parse_data()
        print(len(parsed_links))
        print(len(parsed_content))
    
        buttons = browser.find_elements(By.XPATH,
                                        "//span[contains(@class, 'story__read-more-label') and contains(text(), 'Показать полностью')]") 

        if not buttons:
            browser.execute_script(f"window.scrollBy(0, {scroll_step});") 
            time.sleep(scroll_pause_time)
            continue  

        for button in buttons:
            try:
                
                time.sleep(0.5)  
                button.click()
                time.sleep(0.3)  
            except (ElementClickInterceptedException, StaleElementReferenceException, NoSuchElementException,
                    ElementNotInteractableException):

                continue

        #time.sleep(scroll_pause_time)
        browser.execute_script(f"window.scrollBy(0, {scroll_step});")
        #time.sleep(scroll_pause_time)
        time.sleep(scroll_pause_time)

        new_height = browser.execute_script("return window.pageYOffset;")
        message = browser.find_element(By.XPATH,
                                       "//section[@class='stories-feed__message' and contains(text(), 'Все прочитано!')]")
        if new_height + scroll_step >= browser.execute_script("return document.body.scrollHeight") and message.is_displayed():
            print("Конец страницы достигнут.")
            break


    browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")

    browser.quit()
    return posts


def save_posts_to_csv(posts, filename):
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["Ссылка на пост", "Текст поста"])  
        i=0
        for link, text  in posts:
            i=i+1
            writer.writerow([f"{i}.{link}", text])


author = ""  #имя автора
posts = get_livejournal_posts(author)

if posts:
    save_posts_to_csv(posts, f"posts_pikabu_{author}.csv")
    print(f"Посты успешно сохранены в файл 'posts_pikabu_{author}.csv'")
else:
    print("Не удалось получить посты.")
