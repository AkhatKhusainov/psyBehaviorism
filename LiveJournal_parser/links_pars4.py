import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import time
import os
#from datetime import datetime, date, time

numb=0

def main_links_pars4(name):
    options = Options()
    options.headless = True
    options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"

    author = name  
    driver_path = 'D:\\downloads\\geckodriver-v0.35.0-win64\\geckodriver.exe'
    timeout_seconds = 7  
    start_link = 1  
    restart_interval = 200  
    save_interval = 50  
    #current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = f"posts_lj_{author}.csv"  


    def save_posts_to_csv(posts, filename, append=False):
        mode = 'a' if append and os.path.exists(filename) else 'w'
        with open(filename, mode=mode, newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            if not append or mode == 'w': 
                writer.writerow(["Номер и ссылка на пост", "Текст поста"])
            for n, (link, text) in posts:
                global numb
                numb =numb+1
                clean_text = text.replace('\n', ' ').replace("( Collapse )", "")
                writer.writerow([f"{numb}. {link}", clean_text])


    def initialize_browser():
        try:
            service = Service(executable_path=driver_path)
            browser = webdriver.Firefox(service=service, options=options)
            return browser
        except Exception as e:
            print(f"Ошибка при инициализации браузера: {e}")
            raise

    def parse_page(browser, profile_url):
        posts = []
        try:
            browser.set_page_load_timeout(timeout_seconds)  
            browser.get(profile_url)
            browser.minimize_window()
            time.sleep(0.6)

            checkbox = browser.find_element(By.CSS_SELECTOR, 'input[type="checkbox"]#view-own')
            if not checkbox.is_selected():
                checkbox.click()

            scroll_step = 1000
            browser.execute_script(f"window.scrollBy(0, {scroll_step});")
            time.sleep(0.5)

            expand_buttons = browser.find_elements(By.CLASS_NAME, "ljcut-link-expand")
            for button in expand_buttons:
                try:
                    browser.execute_script("arguments[0].click();", button)
                    time.sleep(0.12)
                    break
                except Exception as e:
                    print(f"Не удалось кликнуть по кнопке: {e}")
                    continue

            # Получение текста поста
            try:
              post_container = browser.find_element(By.CLASS_NAME, 'entry-content')
            #if post_container:

            #else:
            except Exception as e:
                post_container = browser.find_element(By.CLASS_NAME, 'aentry-post__content')
                """if post_container:
                    text = post_container.text
                    posts.append((profile_url, text))
                else:
                    print("Не удалось найти пост.")"""
            text = post_container.text
            posts.append((profile_url, text))

        except Exception as e:
            print(f"Ошибка при парсинге: {e}")
        return posts

    def process_links_with_browser(links, start_index=0, restart_interval=50, save_interval=50):
        posts = []
        browser = initialize_browser()  

        for i, profile_url in enumerate(links[start_index:], start=start_index + 1):
            try:
                print(f"Обработка ссылки {i}: {profile_url}")
                result = parse_page(browser, profile_url)

                if result:
                    posts.extend([(i, url_text) for url_text in result])  
                    print(f"Парсинг страницы {i} завершен.")
                else:
                    print(f"Не удалось получить данные с страницы {i}.")

            except Exception as e:
                print(f"Ошибка при работе с браузером на странице {i}: {e}")

            if (i - start_index) % restart_interval == 0:
                print(f"Перезапуск браузера после обработки {i} ссылок...")
                try:
                    browser.quit()  
                    time.sleep(8)  
                    browser = initialize_browser()  
                    time.sleep(3)
                except Exception as e:
                    print(f"Ошибка при перезапуске браузера: {e}")
                    break

            if i % save_interval == 0 and posts:
                save_posts_to_csv(posts, output_file, append=True)
                print(f"Промежуточное сохранение после {i} ссылок.")
                posts = []  

        browser.quit()  

        if posts:
            save_posts_to_csv(posts, output_file, append=True)

        return posts

    #if __name__ == "__main__":
    posts = []
    with open(f'post_links_{author}.txt', 'r') as file:
        links = [line.strip() for line in file]

    try:
        posts = process_links_with_browser(links, start_index=start_link - 1, restart_interval=restart_interval,
                                           save_interval=save_interval)
    except Exception as e:
        print(f"Общая ошибка в процессе парсинга: {e}")

    print("Парсинг завершен")
    global numb
    numb = 0




