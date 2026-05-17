import pickle

import torch
import numpy as np
import pandas as pd
from IPython.display import display
from fastai.callback.all import EarlyStoppingCallback, SaveModelCallback
from tsai.all import Learner, BCEWithLogitsLossFlat, accuracy_multi, F1ScoreMulti, TSDatasets
from sklearn.metrics import f1_score, precision_score, recall_score

from model import InceptionTimeECG

# Обучение и оценка
def create_learner(dls, config, target_classes, manual_weights_values):
    device = dls.device

    manual_weights = torch.tensor(manual_weights_values).to(device)
    print(f'Используемые веса классов: {target_classes} -> {manual_weights.cpu().numpy()}')

    model = InceptionTimeECG(c_in=12, c_out=5)
    loss_func = BCEWithLogitsLossFlat(pos_weight=manual_weights)

    # Объединяем модель, данные, функцию потерь и метрики для обучения
    learn = Learner(
        dls,
        model,
        metrics=[accuracy_multi, F1ScoreMulti(average='macro')],
        loss_func=loss_func,
        wd=0.1,  # регуляризация
        cbs=[EarlyStoppingCallback(patience=8), SaveModelCallback(fname=config['model_name'])]
    )
    return learn, model


def get_history(learn):
    # Достаем историю обучения из recorder, чтобы потом построить графики
    history = pd.DataFrame(learn.recorder.values)
    metric_names = learn.recorder.metric_names[1:-1]

    if len(history.columns) == len(metric_names):
        history.columns = metric_names
    else:
        n_common = min(len(metric_names), len(history.columns))
        new_cols = list(metric_names[:n_common]) + [f'extra_{i}' for i in range(len(history.columns) - n_common)]
        history.columns = new_cols
    return history


def evaluate_model(probs, targets, thresholds=None):
    # если thresholds не переданы, подбираем их по validation;
    # если переданы, просто применяем готовые пороги
    final_preds = np.zeros_like(probs)
    if thresholds is None:
        # Для каждого класса ищем такой threshold, который даёт лучшую F1-меру
        thresholds = np.zeros(5)
        for i in range(5):
            best_t, best_f1 = 0.5, 0
            for t in np.arange(0.1, 0.9, 0.05):
                score = f1_score(targets[:, i], (probs[:, i] > t).astype(int))
                if score > best_f1:
                    best_f1, best_t = score, t
            thresholds[i] = best_t
            final_preds[:, i] = (probs[:, i] > best_t).astype(int)
        return thresholds, final_preds
    else:
        for i in range(5):
            final_preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
        return final_preds


def count_metrics(targets, preds):
    # Метрики
    return {
        'Accuracy': (preds == targets).mean(),
        'Precision': precision_score(targets, preds, average='macro'),
        'Recall': recall_score(targets, preds, average='macro'),
        'F1 Score': f1_score(targets, preds, average='macro')
    }


def make_result_tables(val_metrics, test_metrics):
    # Таблицы сравнения с RNN, GRU и LSTM
    sander_val = {
        'Model': ['RNN', 'GRU', 'LSTM'],
        'Accuracy': [0.7608, 0.8684, 0.8789],
        'Precision': [0.0125, 0.8228, 0.8350],
        'Recall': [0.0708, 0.5514, 0.5833],
        'F1 Score': [0.0929, 0.6478, 0.6754]
    }
    sander_test = {
        'Model': ['RNN', 'GRU', 'LSTM'],
        'Accuracy': [0.7608, 0.8612, 0.8749],
        'Precision': [0.0125, 0.8202, 0.8247],
        'Recall': [0.0708, 0.5198, 0.5663],
        'F1 Score': [0.0929, 0.6200, 0.6589]
    }

    df_val_final = pd.concat([
        pd.DataFrame(sander_val),
        pd.DataFrame({'Model': ['InceptionTime'], **{k: [v] for k, v in val_metrics.items()}})
    ], ignore_index=True)

    df_test_final = pd.concat([
        pd.DataFrame(sander_test),
        pd.DataFrame({'Model': ['InceptionTime'], **{k: [v] for k, v in test_metrics.items()}})
    ], ignore_index=True)

    return df_val_final, df_test_final


def run_train_and_eval(X, y_encoded, test_idx, dls, config, target_classes, manual_weights_values):
    print('Обучение модели')
    learn, model = create_learner(dls, config, target_classes, manual_weights_values)
    learn.fit_one_cycle(config['epochs'], lr_max=config['lr'])

    history = get_history(learn)

    print('Оценка результатов')
    val_probs, val_targs = learn.get_preds(ds_idx=1)
    best_thresholds, val_preds = evaluate_model(val_probs.numpy(), val_targs.numpy())
    val_metrics = count_metrics(val_targs.numpy(), val_preds)

    test_ds = TSDatasets(X[test_idx], y_encoded[test_idx])
    test_dl = dls.valid.new(test_ds)
    test_probs, test_targs = learn.get_preds(dl=test_dl)
    test_preds = evaluate_model(test_probs.numpy(), test_targs.numpy(), thresholds=best_thresholds)
    test_metrics = count_metrics(test_targs.numpy(), test_preds)

    df_val_final, df_test_final = make_result_tables(val_metrics, test_metrics)

    print('\nТаблица 1. Результаты валидации:')
    display(df_val_final)
    print('\nТаблица 2. Результаты тестирования:')
    display(df_test_final)

    results = {
        'learn': learn,
        'model': model,
        'history': history,
        'best_thresholds': best_thresholds,
        'val_probs': val_probs.numpy(),
        'val_targs': val_targs.numpy(),
        'val_preds': val_preds,
        'test_probs': test_probs.numpy(),
        'test_targs': test_targs.numpy(),
        'test_preds': test_preds,
        'df_val_final': df_val_final,
        'df_test_final': df_test_final,
        # Эти данные нужны для отдельного быстрого построения графиков
        'y_encoded': y_encoded,
        'target_classes': target_classes
    }
    return results


def save_results(results, mlb, config, models_dir, tables_dir, artifacts_dir):
    # Сохраняю всё, что может понадобиться после обучения
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    learn = results['learn']
    model = results['model']
    best_thresholds = results['best_thresholds']

    # 1) Полный fastai Learner
    learner_path = models_dir / f"{config['model_name']}_learner.pkl"
    learn.export(learner_path)

    # 2) Полная PyTorch-модель
    torch.save(model, models_dir / f"{config['model_name']}_full_torch_model.pt")

    # 3) Checkpoint со state_dict и порогами
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'thresholds': best_thresholds
    }, models_dir / f"{config['model_name']}_checkpoint.pt")

    np.save(artifacts_dir / 'best_thresholds.npy', best_thresholds)
    with open(artifacts_dir / 'mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    results['history'].to_csv(tables_dir / 'training_history.csv', index=False)
    results['df_val_final'].to_csv(tables_dir / 'validation_results.csv', index=False)
    results['df_test_final'].to_csv(tables_dir / 'test_results.csv', index=False)

    # Отдельно сохраняю данные для графиков, чтобы потом менять оформление рисунков без повторного обучения модели
    plot_data = {
        'y_encoded': results['y_encoded'],
        'history': results['history'],
        'test_targs': results['test_targs'],
        'test_preds': results['test_preds'],
        'test_probs': results['test_probs'],
        'target_classes': results['target_classes']
    }

    with open(artifacts_dir / 'plot_data.pkl', 'wb') as f:
        pickle.dump(plot_data, f)

    print(f'Модель и результаты сохранены в: {models_dir}')
