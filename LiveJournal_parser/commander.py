
import time
import li_pars7
import links_parser4


names = []  #Список авторов, для парсинга


for name in names:
    print(f"Запуск li_pars7.py с названием: {name}")

    li_pars7.main_li_pars7(name)

    print(f"Запуск links_pars4.py с названием: {name}")
    links_parser4.main_links_pars4(name)

    print(f"Обработка для {name} завершена.\n")
    time.sleep(20)
