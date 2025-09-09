# Documentação Dataset IA

Esse mini projeto foi desenvolvido como forma de atividade avaliativa para a disciplina de Inteligência Artificial(IA).

***Conjunto de Dados Utilizado***: [Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

**Objetivos**: 
- Explorar e analisar o conjunto de dados de qualidade de vinho tinto.
- Comparar modelos de classificação.
- Identificar as features mais relevantes para determinar a qualidade do vinho.

<h3>Importação das Bibliotecas</h3>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, balanced_accuracy_score, classification_report
```

<h3>Upload de Arquivo</h3>

```python
from google.colab import files
uploaded = files.upload()
```

<h3>Verificação de Arquivos no .zip</h3>

```python
import zipfile

with zipfile.ZipFile("wine+quality.zip", "r") as z:
    print(z.namelist())
```

<h3>Carregamento e Verificação do Dataset</h3>

```python
with zipfile.ZipFile("wine+quality.zip") as z:
    with z.open("winequality-red.csv") as f:
        df = pd.read_csv(f, sep=';')

df.head()
df.info()
df.describe()
```

<h3>Pré-processamento</h3>

```python
df["quality_label"] = df["quality"].apply(lambda x: "bom" if x >= 6 else "ruim")

X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]
```

<h3>Divisão de Treino e Teste</h3>

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

<h3>Modelo DecisionTreeClassifier</h3>

```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, dt_pred)
print(f"Acurácia Balanceada: {balanced_acc:.2f}")

print(classification_report(y_test, dt_pred, target_names=[f'Classificação {i}' for i in sorted(y.unique())]))

cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
plt.title('Matriz de Confusão - DecisionTreeClassifier')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.show()
```

<h3>Modelo KNN</h3>

```python
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, knn_pred)
print(f"Acurácia Balanceada: {balanced_acc:.2f}")

print(classification_report(y_test, knn_pred, target_names=[f'Classificação {i}' for i in sorted(y.unique())]))

cm = confusion_matrix(y_test, knn_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
plt.title('Matriz de Confusão - KNN')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.show()
```

<h3>Modelo LogisticRegression</h3>

```python
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, lr_pred)
print(f"Acurácia Balanceada: {balanced_acc:.2f}")

print(classification_report(y_test, lr_pred, target_names=[f'Classificação {i}' for i in sorted(y.unique())]))

cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
plt.title('Matriz de Confusão - LogisticRegression')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.show()
```

<h3>Tarefas</h3>

```python
modelos = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression']
scores = [accuracy_score(y_test, dt_pred), accuracy_score(y_test, knn_pred), accuracy_score(y_test, lr_pred)]

plt.bar(modelos, scores, color=['blue', 'green', 'yellow'])
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia dos Modelos')
plt.show()
```
