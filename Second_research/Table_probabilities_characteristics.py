import anthropic
import json
import pandas as pd
import re
import time

N = 540          
START = 0       
INPUT_FILE = 'Конструкт_все.txt'
OUTPUT_FILE = 'Анализ_с_векторамиCloude3_5_Констр.xlsx'

client = anthropic.Anthropic(
    api_key="claude_key" #ключ для клауд
)
known_causes = set()

def get_class_probabilities_with_causes(text, known_causes):
    prompt = f"""
Ты — эксперт по анализу текстов. Прочитай следующий текст и оцени его по категориям:
- деструктивный
- конструктивный
- информативный

1. Верни вероятности принадлежности к каждой категории (в сумме = 1).
2. Объясни причины классификации.
3. Представь причины в виде JSON-вектора: ключ — причина, значение — от 0 до 3:
   - 0 = не выражено,
   - 1 = слабо выражено,
   - 2 = выражено,
   - 3 = сильно выражено
   Причины могут быть из списка: {list(known_causes)}, но ты должен добавить новые причины, которые могут влиять на выбор категории.

Ответь **только одним JSON-объектом**, со следующей структурой:
{{
  "probabilities": {{
    "деструктивный": 0.0,
    "конструктивный": 0.0,
    "информативный": 0.0
  }},
  "explanation": "...",
  "causes": {{
    "hate": 0,
    "no_solution": 0,
    "facts_only": 0,
    "irony_or_sarcasm": 0,
    "profanity": 0
  }}
}}

Текст:
\"\"\"{text}\"\"\"
"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  
            max_tokens=1500,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response.content[0].text

        match = re.search(r'\{.*"probabilities".*"causes".*\}', content, re.DOTALL)
        if not match:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError("JSON не найден в ответе")

        data = json.loads(match.group(0))

        probs = data.get("probabilities", {})
        explanation = data.get("explanation", "")
        causes = data.get("causes", {})

        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v/total_prob for k, v in probs.items()}

        known_causes.update(causes.keys())

        return {
            "Текст": text,
            "деструктивный": probs.get("деструктивный", 0),
            "конструктивный": probs.get("конструктивный", 0),
            "информативный": probs.get("информативный", 0),
            "Пояснение": explanation,
            "Причины": causes
        }

    except anthropic.RateLimitError:
        print("Превышен лимит requests. Ожидание 60 секунд...")
        time.sleep(60)
        return get_class_probabilities_with_causes(text, known_causes)

    except anthropic.APIError as e:
        raise ValueError(f"Ошибка API Claude: {e}")

all_results = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        texts = [t.strip() for t in content.split('*********') if t.strip()]
except FileNotFoundError:
    print(f"Файл {INPUT_FILE} не найден!")
    exit(1)
except UnicodeDecodeError:
    print("Ошибка кодировки файла. Попробуйте сохранить файл в UTF-8.")
    exit(1)

if START >= len(texts):
    print(f"START ({START}) больше количества текстов ({len(texts)})")
    exit(1)

texts_to_process = texts[START:START+N]
print(f"Найдено {len(texts)} текстов. Обрабатываем {len(texts_to_process)} текстов.")

for idx, text in enumerate(texts_to_process, START+1):
    print(f'Обработка текста #{idx} из {START + len(texts_to_process)}...')

    try:
        row = get_class_probabilities_with_causes(text, known_causes)
        all_results.append(row)

        time.sleep(1)

    except Exception as e:
        print(f'Ошибка при обработке текста #{idx}: {e}')
        all_results.append({
            "Текст": text,
            "деструктивный": None,
            "конструктивный": None,
            "информативный": None,
            "Пояснение": f"Ошибка: {e}",
            "Причины": {}
        })

if all_results:
    df = pd.DataFrame(all_results)

    if "Причины" in df.columns:
        all_causes = set()
        for causes_dict in df["Причины"]:
            if isinstance(causes_dict, dict):
                all_causes.update(causes_dict.keys())

        causes_data = []
        for causes_dict in df["Причины"]:
            if isinstance(causes_dict, dict):
                row = {cause: causes_dict.get(cause, 0) for cause in all_causes}
            else:
                row = {cause: 0 for cause in all_causes}
            causes_data.append(row)

        causes_df = pd.DataFrame(causes_data)
        full_df = pd.concat([df.drop(columns=["Причины"]), causes_df], axis=1)
    else:
        full_df = df
else:
    print("Нет данных для сохранения")
    full_df = pd.DataFrame()

try:
    full_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Результаты сохранены в файл {OUTPUT_FILE}")
    print(f"Обработано текстов: {len(all_results)}")
    if known_causes:
        print(f"Выявлено уникальных причин: {len(known_causes)}")
        print(f"Причины: {', '.join(sorted(known_causes))}")
except Exception as e:
    print(f"Ошибка при сохранении файла: {e}")
    csv_file = OUTPUT_FILE.replace('.xlsx', '.csv')
    full_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Результаты сохранены в CSV файл: {csv_file}")
