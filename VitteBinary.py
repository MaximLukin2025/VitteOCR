import os
import zipfile
import requests

# Ссылка на датасет
dataset_url = "https://www.dropbox.com/scl/fi/mmyjbuglk8iwvybw7qmf5/dataset.zip?rlkey=v10y9pqrqobbmy0tl0zm4l0jl&st=162mmkoi&dl=1"

# Загрузка архива
dataset_path = "dataset.zip"
response = requests.get(dataset_url)
with open(dataset_path, "wb") as f:
    f.write(response.content)

# Распаковка архива
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("Датасет успешно загружен и распакован.")

# Установка Tesseract OCR
!sudo apt-get install tesseract-ocr
!pip install pytesseract

# Импорт библиотек
import pytesseract
from PIL import Image

# Проверка установки
print(pytesseract.get_tesseract_version())

# Установка русского языка
!sudo apt-get install tesseract-ocr-rus

!pip install tensorflow

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

import glob
from tqdm import tqdm  # Для отслеживания прогресса

# OCR извлечение текста
def extract_text(input_folder, label, texts, labels):
    for file_path in tqdm(glob.glob(os.path.join(input_folder, "*.jpg")), desc=f"Processing {label} folder"):
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang="rus")  # Распознаем текст на русском
            texts.append(text)
            labels.append(label)
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

# Пути к папкам
non_vitte_folder = "/content/dataset/dataset/NonVitte"
vitte_folder = "/content/dataset/dataset/Vitte"
texts = []
labels = []

# Извлечение текстов из обеих папок
extract_text(non_vitte_folder, "non_vitte", texts, labels)
extract_text(vitte_folder, "vitte", texts, labels)

print(f"Обработано {len(texts)} изображений.")

# Преобразование текста в признаки
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()
y = [1 if label == "non_vitte" else 0 for label in labels]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# Преобразуем метки в формат NumPy
y_train = np.array(y_train)
y_test = np.array(y_test)

# Функция для построения ROC-кривой
def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Список для хранения метрик и моделей
results = []
best_model = None
best_accuracy = 0

# Эксперимент 1: Простая нейронная сеть с одним скрытым слоем
def experiment_1():
    global best_model, best_accuracy

    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Указание входного размера
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(('Experiment 1', accuracy))
    plot_roc_curve(y_test, y_pred_proba, 'Experiment 1')

    # Сохраняем лучшую модель
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Эксперимент 2: Увеличиваем число нейронов в скрытом слое
def experiment_2():
    global best_model, best_accuracy

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(('Experiment 2', accuracy))
    plot_roc_curve(y_test, y_pred_proba, 'Experiment 2')

    # Сохраняем лучшую модель
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Эксперимент 3: Добавляем еще один скрытый слой
def experiment_3():
    global best_model, best_accuracy

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(('Experiment 3', accuracy))
    plot_roc_curve(y_test, y_pred_proba, 'Experiment 3')

    # Сохраняем лучшую модель
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Запуск экспериментов
plt.figure(figsize=(10, 8))
experiment_1()
experiment_2()
experiment_3()

# Добавляем легенду и подписываем оси
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Experiments')
plt.legend(loc='lower right')
plt.show()

# Отображение результатов
for name, accuracy in results:
    print(f'{name}: Accuracy = {accuracy:.2f}')

# Сохранение лучшей модели и векторизатора
if best_model:
    best_model.save("best_model.h5")
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Лучшая модель сохранена с точностью {best_accuracy:.2f}")

# Скачивание сохраненной модели и векторизатора
from google.colab import files

def download_saved_files():
    files.download("best_model.h5")
    files.download("tfidf_vectorizer.pkl")
    print("Файлы скачаны успешно.")

print("\nСкачивание файлов")
download_saved_files()

from tensorflow.keras.models import load_model

# Загружаем лучшую модель и векторизатор
def load_best_model_and_vectorizer():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    model = load_model("best_model.h5")
    return model, vectorizer

# Классификация нового изображения
def classify_new_image():
    print("Загрузите изображение для классификации")
    uploaded = files.upload()
    for file_name in uploaded.keys():
        try:
            # OCR для нового изображения
            image = Image.open(file_name)
            text = pytesseract.image_to_string(image, lang="rus")

            # Преобразование текста в признаки
            model, vectorizer = load_best_model_and_vectorizer()
            features = vectorizer.transform([text]).toarray()

            # Предсказание
            prediction = model.predict(features).ravel()[0]

            # Интерпретация уверенности
            confidence = max(1e-6, prediction)  # Минимальная уверенность
            confidence = min(1.0 - 1e-6, confidence)  # Максимальная уверенность

            label = "non_vitte" if prediction > 0.5 else "vitte"
            confidence = 1.0 - confidence if label == "sf" else confidence

            print(f"Классификация завершена. Документ: {label} (уверенность: {confidence:.2f})")
        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {e}")

print("Классификация нового изображения")
classify_new_image()

print("Классификация нового изображения")
classify_new_image()
