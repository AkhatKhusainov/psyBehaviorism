import pandas as pd
import numpy as np

def select_best_features(file_path, output_path=None, top_n=20):
    """
    Параметры:
    file_path (str): путь к исходному Excel файлу
    output_path (str): путь для сохранения результата (опционально)
    top_n (int): количество признаков для отбора (по умолчанию 20)

    Возвращает:
    pandas.DataFrame: таблица с отобранными признаками
    """
    print("Загружаем Excel файл...")
    df = pd.read_excel("/content/Анализ_с_векторамиCloude3_5_540.xlsx")

    base_columns = df.columns[:5]  # A, B, C, D, E

    feature_columns = df.columns[5:]  # F и далее

    print(f"Найдено {len(feature_columns)} признаков для анализа")

    feature_scores = {}

    for col in feature_columns:
        zeros_count = (df[col] == 0).sum()
        positive_values_count = df[col].isin([1, 2, 3]).sum()
        total_non_null = df[col].notna().sum()
        zero_ratio = zeros_count / total_non_null if total_non_null > 0 else 1
        positive_ratio = positive_values_count / total_non_null if total_non_null > 0 else 0
        score = positive_ratio - zero_ratio

        feature_scores[col] = {
            'score': score,
            'zeros_count': zeros_count,
            'positive_count': positive_values_count,
            'zero_ratio': zero_ratio,
            'positive_ratio': positive_ratio
        }

    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    top_features = [feature[0] for feature in sorted_features[:top_n]]
    print(f"\nТоп-{top_n} признаков:")
    print("-" * 80)
    print(f"{'Признак':<15} {'Скор':<8} {'Нули':<8} {'1,2,3':<8} {'% нулей':<10} {'% 1,2,3':<10}")
    print("-" * 80)

    for i, feature in enumerate(top_features):
        info = feature_scores[feature]
        print(f"{feature:<15} {info['score']:.3f}    {info['zeros_count']:<8} "
              f"{info['positive_count']:<8} {info['zero_ratio']*100:.1f}%      "
              f"{info['positive_ratio']*100:.1f}%")

    final_columns = list(base_columns) + top_features
    result_df = df[final_columns].copy()

    print(f"\nИтоговая таблица содержит {len(result_df.columns)} столбцов:")
    print(f"- {len(base_columns)} базовых столбцов (A-E)")
    print(f"- {len(top_features)} отобранных признаков")

    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"\nРезультат сохранен в: {output_path}")

    return result_df

if __name__ == "__main__":
    # Укажите путь к вашему Excel файлу
    input_file = "your_data.xlsx"  
    output_file = "selected_features.xlsx"  

    try:
        result = select_best_features(input_file, output_file, top_n=20)

        print("\nОтбор признаков завершен успешно!")

    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден. Проверьте путь к файлу.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def analyze_all_features(file_path):
    """
    Анализирует все признаки и выводит подробную статистику
    """
    df = pd.read_excel(file_path)
    feature_columns = df.columns[5:]  # Начиная с F

    print(f"Анализ всех {len(feature_columns)} признаков:")
    print("=" * 100)

    stats = []
    for col in feature_columns:
        zeros = (df[col] == 0).sum()
        positive = df[col].isin([1, 2, 3]).sum()
        total = df[col].notna().sum()

        stats.append({
            'feature': col,
            'zeros': zeros,
            'positive': positive,
            'total': total,
            'zero_ratio': zeros/total if total > 0 else 0,
            'positive_ratio': positive/total if total > 0 else 0,
            'score': (positive/total - zeros/total) if total > 0 else -1
        })

    stats.sort(key=lambda x: x['score'], reverse=True)

    print(f"{'Место':<6} {'Признак':<15} {'Скор':<8} {'Нули':<8} {'1,2,3':<8} {'% нулей':<10} {'% 1,2,3':<10}")
    print("-" * 100)

    for i, stat in enumerate(stats[:50]):  
        print(f"{i+1:<6} {stat['feature']:<15} {stat['score']:.3f}    "
              f"{stat['zeros']:<8} {stat['positive']:<8} "
              f"{stat['zero_ratio']*100:.1f}%      {stat['positive_ratio']*100:.1f}%")

    return stats
