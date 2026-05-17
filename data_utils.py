import os
import ast
import pickle
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import wfdb
from tqdm.notebook import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from tsai.all import get_ts_dls, TSStandardize

# Подготовка данных
def seed_everything(seed=123):
    # Фиксируем seed, чтобы результаты были воспроизводимыми
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_base_path():
    # Путь к датасету в Kaggle 
    base_path = Path('')
    if os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            if 'ptbxl_database.csv' in files:
                base_path = Path(root)
                break
    else:
        base_path = Path('.')
    return base_path


def aggregate_diagnostic(y_dic, agg_df):
    # Переводим SCP-коды в диагностические суперклассы PTB-XL
    tmp = set()
    for key in y_dic.keys():
        if key in agg_df.index:
            d_class = agg_df.loc[key].diagnostic_class
            if pd.notna(d_class):
                tmp.add(d_class)
    # Возвращаем список классов, потому что у одной ЭКГ может быть несколько меток
    return list(tmp)


def load_signals(df, path, sr):
    # Загружаем сами ECG-сигналы. Если запись не прочиталась, строка остаётся нулевой
    filenames = df['filename_lr'].tolist()
    if len(filenames) == 0:
        return np.zeros((1, 12, sr * 10), dtype=np.float32)

    X = np.zeros((len(filenames), 12, sr * 10), dtype=np.float32)
    for i, fname in enumerate(tqdm(filenames, desc='Загрузка сигналов')):
        try:
            sig, _ = wfdb.rdsamp(str(path / fname))
            # wfdb загружает сигнал как длина x каналы, а для Conv1d нужен формат каналы x длина
            X[i] = sig.T
        except Exception:
            pass
    return X

# Подготовка данных
def prepare_data(config, target_classes, artifacts_dir):
    print('Поиск и загрузка данных')
    base_path = find_base_path()

    if not (base_path / 'ptbxl_database.csv').exists():
        raise FileNotFoundError('Файл ptbxl_database.csv не найден.')

    # Основная таблица PTB-XL с информацией о записях
    df = pd.read_csv(base_path / 'ptbxl_database.csv', index_col='ecg_id')
    
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Таблица с расшифровкой SCP-кодов
    agg_df = pd.read_csv(base_path / 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Для каждой ЭКГ получаем список диагностических суперклассов
    df['diagnostic_superclass'] = df.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
    # Убираем записи без диагностических меток
    df = df[df.diagnostic_superclass.apply(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=target_classes)
    y_encoded = mlb.fit_transform(df['diagnostic_superclass']).astype(np.float32)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / 'mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    # Загружаем сами сигналы ЭКГ
    X = load_signals(df, base_path, config['sampling_rate'])

    # Разбиение датасета по strat_fold: 1-8 — обучение, 9 — валидация, 10 — тест
    train_idx = df[df.strat_fold <= 8].index.tolist()
    val_idx = df[df.strat_fold == 9].index.tolist()
    test_idx = df[df.strat_fold == 10].index.tolist()

    # TSStandardize(by_sample=True) стандартизует каждый сигнал отдельно
    dls = get_ts_dls(
        X,
        y_encoded,
        splits=(train_idx, val_idx),
        bs=config['batch_size'],
        batch_tfms=[TSStandardize(by_sample=True)]
    )

    return df, X, y_encoded, train_idx, val_idx, test_idx, dls, mlb


def make_folders(*folders):
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)


def setup_notebook(seed=123):
    seed_everything(seed)
    warnings.filterwarnings('ignore')
