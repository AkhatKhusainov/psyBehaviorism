# psyBehaviorism
annotation
This study proposes an approach that overcomes these limitations through large language model (LLM) reflection—the ability to analyze its own errors and adjust classification criteria. Unlike traditional methods requiring manual coding and triangulation [6], our technology automates the process while preserving decision transparency.

Behavior prediction is achieved through qualitative analysis of texts at any scale, leveraging large language models' capabilities for interpretation and reflection via iterative dialogue.

Цифровая среда стала важным пространством для изучения человеческого поведения: социальные сети, форумы и блоги фиксируют миллионы текстовых взаимодействий, отражающих эмоции, ценности и социальные динамики. Современные большие языковые модели, такие как Claude, GPT, YandexGPT, активно применяются в социологии и психологии для анализа качественных данных, включая интервью, открытые опросы и публичные дискуссии [1]. Их emergent-способности — качественные скачки в решении задач, которые отсутствуют у меньших моделей [2] — делают их перспективным инструментом для классификации поведения. Однако эта задача сталкивается с двумя серьезными проблемами. Во-первых, контекстная зависимость: одно и то же высказывание может нести разный прагматический смысл. Например, фраза «Власти бездействуют» может быть деструктивной (если цель — нагнетание паники) или конструктивной (если далее следует предложение создать общественный совет), что требует глубокого анализа контекста, аналогично задачам в политических исследованиях [3]. Во-вторых, интерпретируемость: сложные модели машинного обучения, включая LLM, часто работают как «черные ящики», что ограничивает их применение в социальных науках, где объяснимость решений существенно важна [4, 5].  
Данное исследование предлагает подход, преодолевающий эти ограничения через рефлексию больших языковых моделей — способность анализировать собственные ошибки и корректировать критерии классификации. В отличие от традиционных способов, требующих ручного кодирования и триангуляции [6] наша технология должна автоматизировать процесс, сохраняя прозрачность решений.
Прогнозирование поведения осуществляться через квалитативный анализ текстов любого объема с использованием возможностей больших языковых моделей к интерпретации и рефлексии через итеративный диалог.

В данном репозитории представлен пайплайн разработки:

````markdown
```
.
├── LiveJournal_parser/
│   ├── commander.py          # Необходимо ввести имена авторов в список и запустить
│   ├── li_pars7.py
│   └── links_pars4.py
├── Pikabu_parser/
│   └── pikab_pars4.py        # Необходимо ввести имена авторов в список и запустить
├── First_research/
│   ├── Classification/       # Первичная классификация (простой промт)
│   └── Доработка/            # В процессе дорботки
├── Second_research/
│   ├── table_probabilities_features.py  # Создает таблицу распределения вероятностей и признаков (0-3)
│   ├── feature_selection.py             # Сокращает признаки до 20
│   ├── Logistic_regression.py           # Подсчет весов (1000 итераций или условие остановки) необходимо иметь "првильные" one-hot метки для каждого класса
│   ├── Classification_claude.py         # Классификация и сохранение результатов
│   └── Statistics_output.py             # Расчет метрик и визуализация матрицы ошибок
└── README.md
```
````
Литература
1.	Markowitz, D.M. Can generative AI infer thinking style from language? Evaluating the utility of AI as a psychological text analysis tool [Text] / D.M. Markowitz // Behavior Research Methods. – 2024. – V. 56. – P. 3548–3559. 
2.	Wei, J. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models [Text] / J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E.H. Chi, Q.V. Le, D. Zhou //. – 2022.
3.	Linegar, M. Large language models and political science [Text] / M. Linegar, R. Kocielnik, R.M. Alvarez // Frontiers in Political Science. – 2023. – V. 5. 
4.	Bender, E.M. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? [Text] / E.M. Bender, T. Gebru, A. McMillan-Major, S. Shmitchell // Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency. – 2021. – P. 610–623.
5.	Dillon, D. Can AI language models replace human participants? [Text] / D. Dillon, N. Tandon, Y. Gu, K. Gray // Trends in Cognitive Sciences. – 2023. – V. 27. – P. 597–600.
6.	Turner, J.C. Self-categorization theory [Text] / J.C. Turner, K.J. Reynolds // Handbook of theories of social psychology. – 2012. – V 2. – P. 399–417.
