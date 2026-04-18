# NeuroICE Solution

Решение для кейса **NeuroICE** по детекции аномалий в аудиозаписях работы двигателя.

Модель анализирует WAV-файлы, извлекает спектральные и временные признаки и возвращает **anomaly score**:
чем выше значение, тем выше вероятность аномалии.

## Идея решения

В основе решения лежит классический пайплайн для unsupervised anomaly detection:

1. Из аудио извлекаются признаки:
   - MFCC
   - mel-spectrogram
   - spectral centroid
   - spectral rolloff
   - spectral flatness
   - zero crossing rate
   - RMS
   - harmonic/percussive ratio

2. Признаки нормализуются через `RobustScaler`.

3. Размерность дополнительно уменьшается через `PCA`.

4. Для поиска аномалий используется ансамбль моделей:
   - `IsolationForest`
   - `LocalOutlierFactor`
   - `OneClassSVM`

5. Финальный anomaly score получается как агрегированный скор ансамбля с последующей нормализацией.

## Структура проекта

```text
neuroice/
├── solution.py          # основной файл решения
├── model.pkl            # сохранённая модель
├── best_model.pkl       # основная версия ансамбля для инференса
├── scaler.pkl           # scaler + pca + параметры нормализации score
├── notebooks/
│   └── explore.ipynb    # исследование данных и эксперименты
├── scripts/
│   ├── download_data.py
│   └── submit.py
├── pyproject.toml
└── README.md
Основной интерфейс
В solution.py реализован класс PredictionModel:

__init__() загружает обученные артефакты
predict(batch: list[Path]) -> list[float] принимает список WAV-файлов и возвращает anomaly score для каждого файла
Зависимости
Основные библиотеки:

numpy
pandas
scikit-learn
librosa
joblib
Установка
Из папки neuroice:

bash

uv sync
uv pip install librosa joblib
Либо через обычный pip:

bash

pip install numpy pandas scikit-learn librosa joblib
Обучение
Переобучение модели запускается так:

bash

python solution.py
Скрипт:

читает train/train.csv
извлекает признаки из аудио в train/train/
обучает ансамбль
сохраняет:
model.pkl
best_model.pkl
scaler.pkl
Локальная проверка
Пример локального запуска инференса:

bash

python -c "
from pathlib import Path
from solution import PredictionModel

model = PredictionModel()
chunks = sorted((Path('../train/train')).glob('*.wav'))[:5]
print(model.predict(chunks))
"
Submission
Для отправки решения используется:

bash

python scripts/submit.py
Перед этим нужно указать API_KEY в .env.

Особенности текущего решения
Плюсы:

простая и воспроизводимая архитектура
работает без deep learning
быстро загружается
подходит под формат sandbox inference
Ограничения:

качество сильно зависит от качества ручных признаков
модель может быть чувствительна к сдвигу распределения данных
веса ансамбля в базовой версии можно дополнительно улучшать
Возможные улучшения
Направления для дальнейшей работы:

добавить дополнительные аудио-признаки
использовать более устойчивую схему валидации
улучшить подбор весов ансамбля
сравнить разные варианты PCA и scaling
добавить контекст соседних чанков
протестировать более сильные self-supervised подходы