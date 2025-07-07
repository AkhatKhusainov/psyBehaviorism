import pandas as pd
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

gpt_df = pd.read_excel("selected_features.xlsx", header=0)
gpt_df.fillna(0, inplace=True)

prob_cols = gpt_df.columns[1:4]

feature_cols = gpt_df.columns[5:]
X = gpt_df[feature_cols]

gpt_df['Row'] = gpt_df.index + 2  

lines = open("номера_текстовALL.txt", "r", encoding="utf-8").read().splitlines()

data_lines = lines[0:]  

rows = []
for line in data_lines:
    if not line.strip():
        continue

    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        continue
    idx_str, label_str = parts
    try:
        idx = int(idx_str)
        label = ast.literal_eval(label_str)
        if len(label) != 3:
            raise ValueError
    except Exception:
        raise ValueError(f"Не могу распарсить строку: {line}")
    rows.append({'Row': idx, 'dest': label[0], 'constr': label[1], 'info': label[2]})

y_df = pd.DataFrame(rows)

merged = pd.merge(gpt_df, y_df, on='Row', how='inner')

X = merged[feature_cols]
y = merged[['dest','constr','info']]

model = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, solver='lbfgs')
)
model.fit(X, y)

out_rows = []
class_names = ['деструктивный','конструктивный','информативный']
for i, cls in enumerate(class_names):
    lr = model.estimators_[i]
    for feat, coef in zip(feature_cols, lr.coef_[0]):
        out_rows.append({'Класс': cls, 'Признак': feat, 'Вес': coef})
    out_rows.append({'Класс': cls, 'Признак': 'bias', 'Вес': lr.intercept_[0]})

weights_df = pd.DataFrame(out_rows)
weights_df.to_excel("Правильные_веса3.xlsx", index=False)

print("Итоговые веса сохранены в weights.xlsx")
