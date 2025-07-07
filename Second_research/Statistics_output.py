import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_excel_results(excel_file, true_col=2, pred_col=3, sheet_name=0):
    """
    Анализирует результаты классификации из Excel файла

    Args:
        excel_file (str): Путь к Excel файлу
        true_col: Индекс колонки с истинными классами (по умолчанию 2 = колонка C)
        pred_col: Индекс колонки с предсказанными классами (по умолчанию 3 = колонка D)
        sheet_name: Название листа или индекс (по умолчанию 0)
    """

    print(f"📊 Загружаем данные из файла: {excel_file}")

    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"✅ Файл загружен успешно. Строк: {len(df)}")
        print(f"📋 Колонки в файле: {list(df.columns)}")

        if isinstance(true_col, int) and isinstance(pred_col, int):
            print(f"🔍 Используем колонки по индексам: {true_col} и {pred_col}")
            y_true = df.iloc[:, true_col].values
            y_pred = df.iloc[:, pred_col].values
        else:
            print(f"🔍 Используем колонки по названиям: '{true_col}' и '{pred_col}'")
            y_true = df[true_col].values
            y_pred = df[pred_col].values

        mask = pd.notna(y_true) & pd.notna(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        print(f"📈 Анализируем {len(y_true)} записей (после удаления пустых)")

        y_true = [str(cls).lower().strip() for cls in y_true]
        y_pred = [str(cls).lower().strip() for cls in y_pred]

        all_classes = sorted(list(set(y_true + y_pred)))
        print(f"🏷️  Найденные классы: {all_classes}")

        return y_true, y_pred, all_classes, df

    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        return None, None, None, None

def calculate_detailed_metrics(y_true, y_pred, class_names):
    """Вычисляет подробные метрики классификации"""

    overall_accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=class_names,
        average=None,
        zero_division=0
    )

    # Accuracy по классам 
    class_accuracies = []
    for class_name in class_names:
        y_true_binary = [1 if label == class_name else 0 for label in y_true]
        y_pred_binary = [1 if label == class_name else 0 for label in y_pred]

        class_acc = accuracy_score(y_true_binary, y_pred_binary)
        class_accuracies.append(class_acc)

    class_accuracies = np.array(class_accuracies)

    # Макро-усредненные метрики
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    macro_accuracy = np.mean(class_accuracies)

    # Взвешенные метрики
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    return {
        'overall_accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_accuracies': class_accuracies,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def print_results(metrics, class_names):
    """Выводит результаты в красивом формате"""

    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА КЛАССИФИКАЦИИ")
    print("="*80)

    # Общие метрики
    print(f"🎯 Общая точность (Overall Accuracy): {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"📈 Макро F1-score: {metrics['macro_f1']:.4f}")
    print(f"📊 Макро Accuracy: {metrics['macro_accuracy']:.4f} ({metrics['macro_accuracy']*100:.2f}%)")
    print(f"⚖️  Взвешенный F1-score: {metrics['weighted_f1']:.4f}")

    print("\n" + "-"*80)
    print("📋 ПОДРОБНЫЕ МЕТРИКИ ПО КЛАССАМ:")
    print("-"*80)

    print(f"{'Класс':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Поддержка':<12}")
    print("-"*80)

    # Метрики по классам
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {metrics['class_accuracies'][i]:<10.4f} {metrics['precision'][i]:<12.4f} "
              f"{metrics['recall'][i]:<12.4f} {metrics['f1_score'][i]:<12.4f} "
              f"{int(metrics['support'][i]):<12}")

    # Средние значения
    print("-"*80)
    print(f"{'Макро среднее':<20} {metrics['macro_accuracy']:<10.4f} {metrics['macro_precision']:<12.4f} "
          f"{metrics['macro_recall']:<12.4f} {metrics['macro_f1']:<12.4f} "
          f"{int(sum(metrics['support'])):<12}")
    print(f"{'Взвешенное среднее':<20} {'-':<10} {metrics['weighted_precision']:<12.4f} "
          f"{metrics['weighted_recall']:<12.4f} {metrics['weighted_f1']:<12.4f} "
          f"{int(sum(metrics['support'])):<12}")

    print("\n💡 Пояснение к Accuracy по классам:")
    print("   Accuracy для класса = (TP + TN) / (TP + TN + FP + FN)")
    print("   где TP/TN/FP/FN рассчитываются для задачи 'класс vs все остальные'")
    print("   Это показывает, насколько хорошо модель различает данный класс от всех других.")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Строит и показывает матрицу ошибок"""

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Матрица ошибок (Confusion Matrix)', fontsize=16, pad=20)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Матрица ошибок сохранена: {save_path}")

    plt.show()

def save_detailed_report(metrics, class_names, y_true, y_pred, output_file):
    """Сохраняет подробный отчет в Excel"""

    class_metrics_df = pd.DataFrame({
        'Класс': class_names,
        'Accuracy': metrics['class_accuracies'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'Поддержка': metrics['support']
    })

    summary_row = pd.DataFrame({
        'Класс': ['Макро среднее', 'Взвешенное среднее'],
        'Accuracy': [metrics['macro_accuracy'], '-'],
        'Precision': [metrics['macro_precision'], metrics['weighted_precision']],
        'Recall': [metrics['macro_recall'], metrics['weighted_recall']],
        'F1-Score': [metrics['macro_f1'], metrics['weighted_f1']],
        'Поддержка': [sum(metrics['support']), sum(metrics['support'])]
    })

    class_metrics_df = pd.concat([class_metrics_df, summary_row], ignore_index=True)


    general_metrics_df = pd.DataFrame({
        'Метрика': ['Общая точность (Overall Accuracy)', 'Макро Accuracy', 'Макро F1-Score', 'Взвешенный F1-Score'],
        'Значение': [metrics['overall_accuracy'], metrics['macro_accuracy'], metrics['macro_f1'], metrics['weighted_f1']],
        'Процент': [f"{metrics['overall_accuracy']*100:.2f}%",
                   f"{metrics['macro_accuracy']*100:.2f}%",
                   f"{metrics['macro_f1']*100:.2f}%",
                   f"{metrics['weighted_f1']*100:.2f}%"]
    })


    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        general_metrics_df.to_excel(writer, sheet_name='Общие метрики', index=False)
        class_metrics_df.to_excel(writer, sheet_name='Метрики по классам', index=False)
        cm_df.to_excel(writer, sheet_name='Матрица ошибок')

    print(f"💾 Подробный отчет сохранен: {output_file}")

def main():
    """Основная функция"""

    print("🚀 АНАЛИЗАТОР РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ")
    print("="*50)


    excel_file = input("📁 Введите путь к Excel файлу: ").strip()

    print("\nПо умолчанию используются колонки с индексами 2 и 3 (C и D в Excel)")
    custom_cols = input("Хотите изменить? (y/n): ").strip().lower()

    true_col = 2 
    pred_col = 3  

    if custom_cols == 'y':
        print("Введите индексы колонок (A=0, B=1, C=2, D=3, и т.д.)")
        print("Или введите точные названия колонок из файла")

        true_input = input("Истинные классы (колонка): ").strip()
        pred_input = input("Предсказанные классы (колонка): ").strip()

        try:
            true_col = int(true_input)
        except ValueError:
            true_col = true_input

        try:
            pred_col = int(pred_input)
        except ValueError:
            pred_col = pred_input

    y_true, y_pred, class_names, df = analyze_excel_results(excel_file, true_col, pred_col)

    if y_true is None:
        return

    metrics = calculate_detailed_metrics(y_true, y_pred, class_names)

    print_results(metrics, class_names)

    print("\n" + "="*50)
    print("ДОПОЛНИТЕЛЬНЫЕ ОПЦИИ:")

    show_cm = input("📊 Показать матрицу ошибок? (y/n): ").strip().lower()
    if show_cm == 'y':
        plot_confusion_matrix(y_true, y_pred, class_names)

    save_report = input("💾 Сохранить подробный отчет в Excel? (y/n): ").strip().lower()
    if save_report == 'y':
        output_file = input("Введите имя файла для сохранения (например, report.xlsx): ").strip()
        if not output_file.endswith('.xlsx'):
            output_file += '.xlsx'
        save_detailed_report(metrics, class_names, y_true, y_pred, output_file)

    show_sklearn = input("📋 Показать sklearn classification_report? (y/n): ").strip().lower()
    if show_sklearn == 'y':
        print("\n" + "="*50)
        print("SKLEARN CLASSIFICATION REPORT:")
        print("="*50)
        print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
