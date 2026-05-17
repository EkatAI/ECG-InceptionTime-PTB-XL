
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


# Цвета классов: классическая палитра matplotlib
# Эти же цвета используются на гистограмме и на PR-кривых
class_colors = {
    'CD': '#1f77b4',      # синий
    'HYP': '#ff7f0e',     # оранжевый
    'MI': '#2ca02c',      # зелёный
    'NORM': '#9467bd',    # фиолетовый
    'STTC': '#d62728'     # красный
}

# Стили линий для чёрно-белой печати.
line_styles = {
    'CD': '-',
    'HYP': '--',
    'MI': ':',
    'NORM': '-.',
    'STTC': (0, (8, 3, 1, 3, 1, 3, 1, 3))
}


def to_numpy(x):
    # Чтобы функции работали и с numpy-массивами, и с torch-тензорами
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_report_font():
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if 'Times New Roman' in available_fonts:
        return 'Times New Roman'

    return 'DejaVu Serif'

def setup_plot_style():
    # Единый стиль для всех рисунков
    report_font = get_report_font()

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [report_font, 'Times New Roman', 'DejaVu Serif'],
        'font.size': 12,

        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,

        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,

        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        'axes.unicode_minus': False,

        # Тонкая ненавязчивая штриховка для гистограммы
        'hatch.linewidth': 0.4,

        # Для векторных файлов текст остаётся текстом
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })


def save_fig(fig, name, figures_dir):
    # PNG — для отчёта, SVG — векторный вариант
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / f'{name}.png', dpi=600, bbox_inches='tight')
    fig.savefig(figures_dir / f'{name}.svg', bbox_inches='tight')

def plot_class_distribution(y_encoded, target_classes, figures_dir):
    setup_plot_style()

    y_encoded = to_numpy(y_encoded)
    counts = y_encoded.sum(axis=0)

    # Очень лёгкая штриховка для различимости в ч/б
    light_hatches = {
        'CD': '/',
        'HYP': '\\',
        'MI': '.',
        'NORM': 'o',
        'STTC': '-'
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        target_classes,
        counts,
        color=[class_colors[cls] for cls in target_classes],
        edgecolor='#444444',
        linewidth=0.8,
        alpha=0.9
    )

    for bar, cls, count in zip(bars, target_classes, counts):
        bar.set_hatch(light_hatches[cls])

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(int(count)),
            ha='center',
            va='bottom',
            fontsize=10
        )
    # Чтобы числа над столбцами не прижимались к границе рисунка
    ax.set_ylim(0, max(counts) * 1.08)
    
    ax.set_title('Распределение диагностических суперклассов')
    ax.set_xlabel('Диагностический суперкласс')
    ax.set_ylabel('Количество записей')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, 'class_distribution', figures_dir)
    plt.show()

def plot_confusion_matrices(test_targs, test_preds, target_classes, figures_dir):
    setup_plot_style()

    test_targs = to_numpy(test_targs)
    test_preds = to_numpy(test_preds)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, cls in enumerate(target_classes):
        cm = confusion_matrix(
            test_targs[:, i],
            test_preds[:, i],
            normalize='true'
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=axes[i],
            cbar=False,
            vmin=0,
            vmax=1
        )

        # Подписи
        axes[i].set_title(f'Матрица ошибок: {cls}')
        axes[i].set_ylabel('Истинный класс')
        axes[i].set_xlabel('Предсказанный класс')

    fig.delaxes(axes[5])

    plt.tight_layout()
    save_fig(fig, 'confusion_matrices', figures_dir)
    plt.show()


def plot_loss(history, figures_dir):
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    if 'train_loss' in history.columns:
        ax.plot(
            history.index + 1,
            history['train_loss'],
            label='Ошибка обучения (Train)',
            color='#1f77b4',
            linestyle='-',
            linewidth=2.2
        )

    if 'valid_loss' in history.columns:
        ax.plot(
            history.index + 1,
            history['valid_loss'],
            label='Ошибка валидации (Val)',
            color='#ff7f0e',
            linestyle='--',
            linewidth=2.2
        )

    ax.set_title('График функции потерь (Loss)')
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Значение ошибки')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'loss_history', figures_dir)
    plt.show()

def plot_accuracy(history, figures_dir):
    acc_cols = [c for c in history.columns if 'accuracy' in c]

    if not acc_cols:
        return

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        history.index + 1,
        history[acc_cols[0]],
        color='#2ca02c',
        linestyle='-',
        linewidth=2.2,
        label='Точность (Accuracy)'
    )

    ax.set_title('Точность модели на валидации')
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Точность')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'accuracy_history', figures_dir)
    plt.show()
def plot_precision_recall_curves(test_targs, test_probs, target_classes, figures_dir):
    setup_plot_style()

    test_targs = to_numpy(test_targs)
    test_probs = to_numpy(test_probs)

    fig, ax = plt.subplots(figsize=(9, 8))

    lines_info = []

    for i, cls in enumerate(target_classes):
        precision, recall, _ = precision_recall_curve(
            test_targs[:, i],
            test_probs[:, i]
        )

        pr_auc = average_precision_score(
            test_targs[:, i],
            test_probs[:, i]
        )

        n_points = min(150, len(recall))
        indices = np.linspace(0, len(recall) - 1, n_points).astype(int)
        indices = np.unique(indices)

        r_plot = recall[indices]
        p_plot = precision[indices]

        line, = ax.plot(
            r_plot,
            p_plot,
            color=class_colors[cls],
            linestyle=line_styles[cls],
            linewidth=2.5
        )

        label = f'{cls} (AUC = {pr_auc:.3f})'
        lines_info.append((pr_auc, line, label))

    # Легенду сортируем по AUC по убыванию
    lines_info.sort(key=lambda x: x[0], reverse=True)

    sorted_handles = [item[1] for item in lines_info]
    sorted_labels = [item[2] for item in lines_info]

    ax.set_xlabel('Полнота (Recall)')
    ax.set_ylabel('Точность (Precision)')
    ax.set_title('Кривые точности-полноты (Precision-Recall)', pad=15)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.05])

    ax.grid(True, linestyle=':', alpha=0.45, color='black')

    leg = ax.legend(
        handles=sorted_handles,
        labels=sorted_labels,
        loc='lower left',
        frameon=True,
        fontsize=11,
        handlelength=6.0
    )

    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout()
    save_fig(fig, 'precision_recall_curves', figures_dir)
    plt.show()

def plot_confusion_matrices_individual(test_targs, test_preds, target_classes, figures_dir):
    setup_plot_style()

    test_targs = to_numpy(test_targs)
    test_preds = to_numpy(test_preds)

    for i, cls in enumerate(target_classes):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))

        cm = confusion_matrix(
            test_targs[:, i],
            test_preds[:, i],
            normalize='true'
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            cbar=True,
            vmin=0,
            vmax=1,
            xticklabels=['0', '1'],
            yticklabels=['0', '1']
        )

        ax.set_title(f'Матрица ошибок: {cls}')
        ax.set_ylabel('Истинный класс')
        ax.set_xlabel('Предсказанный класс')

        plt.tight_layout()
        save_fig(fig, f'confusion_matrix_{cls}', figures_dir)
        plt.show()
        
def plot_all_figures(y_encoded, history, test_targs, test_preds, test_probs, target_classes, figures_dir):
    plot_class_distribution(y_encoded, target_classes, figures_dir)
    plot_confusion_matrices(test_targs, test_preds, target_classes, figures_dir)
    plot_confusion_matrices_individual(test_targs, test_preds, target_classes, figures_dir)
    plot_loss(history, figures_dir)
    plot_accuracy(history, figures_dir)
    plot_precision_recall_curves(test_targs, test_probs, target_classes, figures_dir)
