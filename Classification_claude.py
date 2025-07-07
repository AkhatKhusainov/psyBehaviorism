import anthropic
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import re

INPUT_FILE = 'СмешанныеALL.txt'
LABELS_FILE = 'номера_текстовALL.txt'  
OUTPUT_FILE = 'результаты_классификации.xlsx'
N_TEXTS = 540  

client = anthropic.Anthropic(
    api_key=claude_api  #API ключ
)

CLASS_MAPPING = {
    'деструктивный': [1, 0, 0],
    'конструктивный': [0, 1, 0],
    'информативный': [0, 0, 1]
}

CLASS_NAMES = ['деструктивный', 'конструктивный', 'информативный']

def classify_text(text, custom_prompt=""):
    """
    Классифицирует текст с помощью Claude API

    Args:
        text (str): Текст для классификации
        custom_prompt (str): Пользовательский промпт

    Returns:
        str: Класс текста или None при ошибке
    """

    base_prompt = f"""
{custom_prompt}

Текст для анализа:
\"\"\"{text}\"\"\"

Ответь **только одним JSON-объектом** со следующей структурой:
{{"класс": "деструктивный/конструктивный/информативный"}}
"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  
            max_tokens=100,
            temperature=0.1,
            messages=[{"role": "user", "content": base_prompt}]
        )

        content = response.content[0].text.strip()

        
        json_match = re.search(r'\{[^}]*"класс"[^}]*\}', content)
        if json_match:
            data = json.loads(json_match.group(0))
            predicted_class = data.get("класс", "").lower()

           
            if "деструктив" in predicted_class:
                return "деструктивный"
            elif "конструктив" in predicted_class:
                return "конструктивный"
            elif "информатив" in predicted_class:
                return "информативный"

        return None

    except Exception as e:
        print(f"Ошибка при классификации: {e}")
        return None

def load_texts(filename):
    """Загружает тексты из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            texts = [t.strip() for t in f.read().split('*********') if t.strip()]
        return texts
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return []

def load_labels(filename):
    """Загружает правильные метки из файла в формате: номер [one-hot вектор]"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
       
            match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', line)
            if match:
                vector = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
                labels.append(vector)

        return labels
    except Exception as e:
        print(f"Ошибка при чтении файла меток {filename}: {e}")
        return []

def text_to_onehot(class_name):
    """Преобразует название класса в one-hot encoding"""
    return CLASS_MAPPING.get(class_name, [0, 0, 0])

def onehot_to_class_name(vector):
    """Преобразует one-hot вектор в название класса"""
    if vector == [1, 0, 0]:
        return "деструктивный"
    elif vector == [0, 1, 0]:
        return "конструктивный"
    elif vector == [0, 0, 1]:
        return "информативный"
    else:
        return "неопределенный"

def calculate_metrics(y_true, y_pred):
    """Вычисляет метрики качества классификации"""

    y_true_idx = [np.argmax(label) for label in y_true]
    y_pred_idx = [np.argmax(label) for label in y_pred]

    overall_accuracy = accuracy_score(y_true_idx, y_pred_idx)

    class_accuracies = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_true = [1 if idx == i else 0 for idx in y_true_idx]
        class_pred = [1 if idx == i else 0 for idx in y_pred_idx]
        class_acc = accuracy_score(class_true, class_pred)
        class_accuracies[class_name] = class_acc

    return overall_accuracy, class_accuracies

def main():
    print("🚀 Начинаем классификацию текстов...")

    texts = load_texts(INPUT_FILE)
    true_labels = load_labels(LABELS_FILE)

    if not texts:
        print("❌ Не удалось загрузить тексты")
        return

    if not true_labels:
        print("❌ Не удалось загрузить метки")
        return

    if len(texts) != len(true_labels):
        print(f"⚠️  Количество текстов ({len(texts)}) не совпадает с количеством меток ({len(true_labels)})")
        min_len = min(len(texts), len(true_labels))
        texts = texts[:min_len]
        true_labels = true_labels[:min_len]

    if N_TEXTS is not None and N_TEXTS > 0:
        texts = texts[:N_TEXTS]
        true_labels = true_labels[:N_TEXTS]
        print(f"🔢 Ограничиваем обработку до {N_TEXTS} текстов")

    print(f"📊 Обрабатываем {len(texts)} текстов")

    custom_prompt = """
Информативный тип: тексты, содержащие аналитику, статистические данные, НЕ содержащие призывов к каким-либо действиям.

Деструктивный тип: тексты, содержащие неконструктивную критику, явно и неявно призывающие к необдуманным разрушительным действиям, оскорбляющие отдельные социальные группы, направленные на нагнетание беспокойства и паники.

Конструктивный тип: тексты включают материалы, направленные на формирование спокойного, позитивного или критического отношения к описываемым событиям, явлениям или предметам. Эти тексты характеризуются разнообразными особенностями и целевыми направленностями. Они призывают к созидательным действиям, побуждают к осмыслению ситуации и критическому анализу информации, формируют определенное отношение к происходящим событиям, вызывают интерес к теме и способствуют расширению кругозора читателя. В лексическом плане такие тексты содержат успокаивающе-ободряющую, эмоционально окрашенную лексику, используют оценочные суждения, метафоры и сравнения, могут включать элементы разговорного стиля. Стилистически они часто прибегают к иронии, юмору или сарказму для снижения напряженности или усиления эффекта, могут содержать элементы самоиронии, применяют как разговорный, так и аналитический стиль изложения. Практическая направленность проявляется в предложении конкретных советов, рекомендаций или решений проблем, включении личных рекомендаций автора и обучающих материалов. Аналитический компонент выражается через включение статистических данных, прогнозов с позитивным или нейтральным подтекстом, сравнительного анализа и объяснения сложных явлений простым языком. Субъективный аспект проявляется в выражении личного мнения автора, описании его опыта и размышлениях о будущем. Патриотический и социальный аспекты могут включать элементы, вызывающие чувство гордости за страну и уважение к определенным группам. Интеллектуальная стимуляция достигается через побуждение к размышлению, самопознанию, предложение альтернативных точек зрения и новых интерпретаций известных фактов. Эмоциональное воздействие оказывается через элементы мотивации, вдохновения и поддержки читателя. Критический подход в этих текстах направлен на улучшение ситуации, предлагая конструктивные решения проблем. Жанровое разнообразие может включать элементы рецензий, обзоров, анализ культурных явлений и исторических событий. Важной чертой является сбалансированность: признание проблем сочетается с обещанием их решения, формируется целостное понимание ситуации без нагнетания паники и призывов к деструктивным действиям. Тексты конструктивного типа могут сочетать различные элементы из вышеперечисленных характеристик, адаптируясь к конкретной теме, целевой аудитории и контексту.
   Обязательно учитывай слудующие веса:
деструктивный	factual_presentation	-0,034853278
деструктивный	facts_only	-0,202618163
деструктивный	clear_structure	-0,451201609
деструктивный	irony_or_sarcasm	0,010749291
деструктивный	specific_examples	0,121033548
деструктивный	social_criticism	0,265700991
деструктивный	educational_content	-0,203548961
деструктивный	personal_experience	0,134799274
деструктивный	balanced_view	-0,181456765
деструктивный	negative_generalizations	0,541024425
деструктивный	solution_oriented	-0,241887558
деструктивный	statistical_data	-0,180872435
деструктивный	negative_emotions	0,891916655
деструктивный	constructive_analysis	-0,120867823
деструктивный	social_commentary	-0,144009235
деструктивный	objective_tone	-0,005724606
деструктивный	system_criticism	0,400172319
деструктивный	political_analysis	-0,047437309
деструктивный	positive_tone	-1,231834051
деструктивный	cynical_tone	0,070493336
деструктивный	bias	-0,952973591
конструктивный	factual_presentation	-0,576339789
конструктивный	facts_only	-0,528400994
конструктивный	clear_structure	0,105531383
конструктивный	irony_or_sarcasm	0,081179477
конструктивный	specific_examples	-0,03758022
конструктивный	social_criticism	-0,210361567
конструктивный	educational_content	-0,373359809
конструктивный	personal_experience	-0,096930384
конструктивный	balanced_view	0,309301922
конструктивный	negative_generalizations	-0,564663629
конструктивный	solution_oriented	0,334970556
конструктивный	statistical_data	-0,21337158
конструктивный	negative_emotions	-0,681503604
конструктивный	constructive_analysis	0,060344665
конструктивный	social_commentary	0,334124066
конструктивный	objective_tone	-0,197297045
конструктивный	system_criticism	-0,073591643
конструктивный	political_analysis	-0,526132702
конструктивный	positive_tone	0,903643077
конструктивный	cynical_tone	-0,043599614
конструктивный	bias	0,268718818
информативный	factual_presentation	0,557649492
информативный	facts_only	0,614700835
информативный	clear_structure	0,142095424
информативный	irony_or_sarcasm	-0,007084199
информативный	specific_examples	-0,040168742
информативный	social_criticism	-0,254174574
информативный	educational_content	0,474147018
информативный	personal_experience	-0,021974295
информативный	balanced_view	-0,159764384
информативный	negative_generalizations	-0,117612586
информативный	solution_oriented	-0,219157958
информативный	statistical_data	0,353563122
информативный	negative_emotions	-0,860173487
информативный	constructive_analysis	0,037437032
информативный	social_commentary	-0,302848374
информативный	objective_tone	0,107106275
информативный	system_criticism	-0,470661942
информативный	political_analysis	0,516776322
информативный	positive_tone	-0,556881764
информативный	cynical_tone	-0,03120439
информативный	bias	-1,500200134
    """

    # Или оставьте интерактивный ввод:
    # print("\n" + "="*50)
    # print("Введите ваш промпт для классификации:")
    # custom_prompt = input().strip()
    print("\n" + "="*50)
    print("Введите ваш промпт для классификации:")
    print("(или нажмите Enter для использования базового)")
    custom_prompt = input().strip()

    if not custom_prompt:
        custom_prompt = "Классифицируй следующий текст как деструктивный, конструктивный или информативный."

    print(f"\n📊 Будет обработано текстов: {len(texts)}")
    print(f"🎯 Используется промпт: {custom_prompt[:100]}...")

    predictions = []
    results_detailed = []

    for i, text in enumerate(texts, 1):
        print(f"📝 Обрабатываем текст {i}/{len(texts)}...", end=" ")

        predicted_class = classify_text(text, custom_prompt)
        predictions.append(predicted_class)

        results_detailed.append({
            'Номер': i,
            'Текст': text[:100] + "..." if len(text) > 100 else text,
            'Истинный_класс': onehot_to_class_name(true_labels[i-1]),
            'Предсказанный_класс': predicted_class,
            'Корректно': predicted_class == onehot_to_class_name(true_labels[i-1]) if predicted_class else False
        })

        if predicted_class:
            print(f"✅ {predicted_class}")
        else:
            print("❌ Ошибка")

        time.sleep(1)

    y_true = true_labels  
    y_pred = [text_to_onehot(pred) if pred else [0, 0, 0] for pred in predictions]

    overall_acc, class_accuracies = calculate_metrics(y_true, y_pred)

    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("="*60)
    print(f"🎯 Общая точность: {overall_acc:.3f} ({overall_acc*100:.1f}%)")
    print("\n📈 Точность по классам:")

    for class_name, acc in class_accuracies.items():
        print(f"   {class_name:15} : {acc:.3f} ({acc*100:.1f}%)")

    df_results = pd.DataFrame(results_detailed)
  
    summary_data = {
        'Метрика': ['Общая точность'] + [f'Точность - {cls}' for cls in CLASS_NAMES],
        'Значение': [overall_acc] + list(class_accuracies.values()),
        'Процент': [f"{overall_acc*100:.1f}%"] + [f"{acc*100:.1f}%" for acc in class_accuracies.values()]
    }
    df_summary = pd.DataFrame(summary_data)

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Детальные результаты', index=False)
        df_summary.to_excel(writer, sheet_name='Сводка', index=False)

    print(f"\n💾 Результаты сохранены в файл: {OUTPUT_FILE}")
    print(f"📋 Всего обработано текстов: {len(texts)}")
    print(f"✅ Успешно классифицировано: {sum(1 for p in predictions if p is not None)}")
    print(f"❌ Ошибок классификации: {sum(1 for p in predictions if p is None)}")

if __name__ == "__main__":
    main()
