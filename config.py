from pathlib import Path

# Основные настройки эксперимента
config = {
    'seed': 123,
    'lr': 0.0001,
    'batch_size': 64,
    'epochs': 25,
    'sampling_rate': 100,
    'model_name': 'InceptionTimeECG'
}

# Порядок классов
target_classes = sorted(['CD', 'HYP', 'MI', 'NORM', 'STTC'])
manual_weights_values = [1.0, 2.0, 2.0, 1.0, 1.5]

# Путь для сохранения результатов
WORK_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
MODELS_DIR = WORK_DIR / 'models'
FIGURES_DIR = WORK_DIR / 'figures'
TABLES_DIR = WORK_DIR / 'tables'
ARTIFACTS_DIR = WORK_DIR / 'artifacts'

# Цвета классов
class_colors = {
    'CD': '#1f77b4',    # синий
    'HYP': '#ff7f0e',   # оранжевый
    'MI': '#2ca02c',    # зелёный
    'NORM': '#9467bd',  # фиолетовый
    'STTC': '#d62728'   # красный
}

# Дополнительные стили нужны, чтобы линии отличались при чёрно-белой печати
line_styles = {
    'CD': '-',
    'HYP': '--',
    'MI': ':',
    'NORM': '-.',
    'STTC': (0, (8, 3, 1, 3, 1, 3, 1, 3))
}
