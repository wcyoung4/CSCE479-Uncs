import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class SaveModelEpoch(tf.keras.callbacks.Callback):
  def __init__(self,save_str, epoch_interval):
      super().__init__()
      self.save_str=save_str
      self.epoch_interval=epoch_interval
      self.count=0

  def on_epoch_end(self, epoch, logs=None):
      if (epoch + 1) % self.epoch_interval == 0:
          save_str_count=self.save_str+str(self.count)+'.keras'
          self.model.save(save_str_count)
          self.count+=1


def run_name(arch_name, dropout_rate, lr):
    return '%s_lr%.0e_do%s' % (arch_name, lr, dropout_rate)


def find_run(results, name):
    for row in results:
        if row['run'] == name:
            return row


def accuracy_ci(accuracy, n):
    half = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
    return half, accuracy - half, accuracy + half


def error_difference_ci(label, row_a, row_b, correctness, key='test_accuracy'):
    correct_a = np.asarray(correctness[row_a['run']][key], dtype=bool)
    correct_b = np.asarray(correctness[row_b['run']][key], dtype=bool)
    if len(correct_a) != len(correct_b):
        raise ValueError('Models must use the same test examples')

    err_a = (~correct_a).astype(float)
    err_b = (~correct_b).astype(float)
    per_image_diff = err_a - err_b
    diff = float(np.mean(per_image_diff))
    half = float(1.96 * np.std(per_image_diff, ddof=1) / np.sqrt(len(per_image_diff)))
    return {'comparison': label,
            'metric': key,
            'model_a': row_a['run'],
            'model_b': row_b['run'],
            'error_a': float(np.mean(err_a)),
            'error_b': float(np.mean(err_b)),
            'difference': diff,
            'ci_low': diff - half,
            'ci_high': diff + half,
            'significant': diff - half > 0 or diff + half < 0}


def top_k_accuracy(probs, trues, k=5):
    top_k = np.argsort(-probs, axis=1)[:, :k]
    return np.mean(np.any(top_k == trues[:, None], axis=1))


def confusion_matrix(trues, preds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    return cm


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    columns = list(rows[0].keys())
    cells = []
    for row in rows:
        cells.append(['%.4f' % row[c] if isinstance(row[c], float) else str(row[c])
                      for c in columns])
    widths = [max([len(c)] + [len(cell[i]) for cell in cells])
              for i, c in enumerate(columns)]
    header = '  '.join(c.ljust(w) for c, w in zip(columns, widths))
    print(header)
    print('-' * len(header))
    for cell in cells:
        print('  '.join(c.ljust(w) for c, w in zip(cell, widths)))


#architectures
def _layer_details(layer):
    cfg = layer.get_config()
    kind = layer.__class__.__name__
    if kind == 'Conv2D':
        return '%d filters, %dx%d, %s' % (cfg['filters'], cfg['kernel_size'][0], cfg['kernel_size'][1], cfg['padding'])
    if kind == 'Dense':
        return '%d units, %s' % (cfg['units'], layer.activation.__name__)
    if kind == 'MaxPooling2D':
        return '%dx%d pool' % tuple(cfg['pool_size'])
    if kind == 'Dropout':
        return 'rate %s' % cfg['rate']
    if kind == 'Activation':
        return layer.activation.__name__
    if kind == 'Rescaling':
        return 'x / 255'
    if kind == 'RandomFlip':
        return cfg['mode'] + ' (train only)'
    if kind == 'RandomTranslation':
        return '+/- %s shift (train only)' % cfg['height_factor']
    return ''


def architecture_table(model):
    rows = []
    for i, layer in enumerate(model.layers):
        rows.append({'#': i + 1,
                     'layer': layer.__class__.__name__,
                     'details': _layer_details(layer),
                     'output_shape': str(tuple(layer.output.shape[1:])),
                     'params': layer.count_params()})
    return rows


def plot_architecture_table(rows, path, title):
    columns = list(rows[0].keys())
    cells = [[str(row[c]) for c in columns] for row in rows]
    total = sum(row['params'] for row in rows)
    cells.append(['', 'Total', '', '', str(total)])
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(cells) + 0.9))
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=columns, loc='center', cellLoc='left',
                     colWidths=[0.05, 0.22, 0.35, 0.22, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        if r == 0 or r == len(cells):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6eef7')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


#FIGURES
def plot_confusion_matrix(cm, path, class_names, normalize=True, annotate=True, title='Confusion matrix'):
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    size = 8.5 if n <= 20 else 16
    tick_size = 8 if n <= 20 else 4
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=data.max())
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=90 if n > 20 else 45, ha='center' if n > 20 else 'right', fontsize=tick_size)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=tick_size)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title + (' (row-normalized)' if normalize else ''))
    if annotate:
        threshold = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if normalize:
                    text = '{:.0%}'.format(data[i, j]) if data[i, j] >= 0.005 else ''
                else:
                    text = str(int(cm[i, j]))
                ax.text(j, i, text, ha='center', va='center', fontsize=6,
                        color='white' if data[i, j] > threshold else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histories(histories, path):
    """Training / validation curves for every run in the grid."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    cmap = plt.get_cmap('tab10')
    panels = [('loss', 'Loss', 'Cross-entropy'),
              ('accuracy', 'Top-1 accuracy', 'Accuracy'),
              ('top5_accuracy', 'Top-5 accuracy', 'Accuracy')]
    for idx, (name, hist) in enumerate(histories.items()):
        color = cmap(idx % 10)
        epochs = range(1, len(hist['loss']) + 1)
        for ax, (key, _, _) in zip(axes, panels):
            ax.plot(epochs, hist[key], color=color, alpha=0.35)
            ax.plot(epochs, hist['val_' + key], color=color, label=name)
    for ax, (_, title, ylabel) in zip(axes, panels):
        ax.set_title(title + ' (faded = train, solid = validation)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
    axes[2].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction_grid(images, trues, preds, path, class_names,
                         indices=None, n=24, ncols=6, title='Test predictions'):
    """Grid of test images titled with predicted and true class, wrong = red."""
    if indices is None:
        indices = np.arange(min(n, len(images)))
    indices = np.asarray(indices)[:n]
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.3 * nrows),
                             squeeze=False)
    flat = np.ravel(axes)
    for ax, idx in zip(flat, indices):
        ax.imshow(images[idx])
        correct = preds[idx] == trues[idx]
        ax.set_title('pred: {}\ntrue: {}'.format(class_names[preds[idx]],
                                                 class_names[trues[idx]]),
                     fontsize=8, color='green' if correct else 'red')
        ax.axis('off')
    for ax in flat[len(indices):]:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def most_confident_errors(probs, preds, trues, n=24):
    """Indices of the misclassifications the model was most confident about."""
    wrong = np.flatnonzero(preds != trues)
    if len(wrong) == 0:
        return wrong
    confidence = probs[wrong, preds[wrong]]
    return wrong[np.argsort(-confidence)][:n]


#  integrated gradients
def integrated_gradients(model, image, target, baseline=None, m_steps=64, batch_size=32):
    """Integrated Gradients for one image toward one class."""
    image = tf.cast(image, tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(image)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)

    grads = []
    for start in range(0, m_steps + 1, batch_size):
        alpha = alphas[start:start + batch_size][:, None, None, None]
        x = baseline[None] + alpha * (image - baseline)[None]
        with tf.GradientTape() as tape:
            tape.watch(x)
            probs = model(x, training=False)
            score = probs[:, target]
        grads.append(tape.gradient(score, x))
    grads = tf.concat(grads, axis=0)

    # trapezoid rule for the path integral
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)
    attributions = (image - baseline) * avg_grads

    ends = model(tf.stack([baseline, image]), training=False)[:, target]
    delta = float(ends[1] - ends[0])
    return attributions.numpy(), delta


def select_ig_examples(probs, preds, trues, cls):
    """For one true class pick the most confident correct, least confident correct, and most confident wrong prediction so they can be compared."""
    picks = []
    members = np.flatnonzero(trues == cls)
    right = members[preds[members] == cls]
    wrong = members[preds[members] != cls]
    if len(right) > 0:
        conf = probs[right, cls]
        picks.append(('high-confidence correct', right[np.argmax(conf)]))
        if len(right) > 1:
            picks.append(('low-confidence correct', right[np.argmin(conf)]))
    if len(wrong) > 0:
        conf = probs[wrong, preds[wrong]]
        picks.append(('high-confidence wrong', wrong[np.argmax(conf)]))
    return picks


def _prep_attribution(attr, clip=99):
    s = np.abs(attr).sum(axis=-1)
    return np.clip(s / max(np.percentile(s, clip), 1e-12), 0, 1)


def plot_ig_grid(images, entries, path, class_names, title):
    """One column per example."""
    fig, axes = plt.subplots(4, len(entries), figsize=(2.3 * len(entries), 9.2),
                             squeeze=False)
    for col, e in enumerate(entries):
        img = images[e['index']]
        attr_pred = _prep_attribution(e['ig_pred'])
        attr_true = _prep_attribution(e['ig_true'])
        correct = e['pred'] == e['true']
        axes[0, col].imshow(img)
        axes[0, col].set_title('{}\npred: {} ({:.0%})\ntrue: {}'.format(
            e['criterion'], class_names[e['pred']], e['confidence'], class_names[e['true']]),
            fontsize=7, color='green' if correct else 'red')
        axes[1, col].imshow(attr_pred, cmap='inferno')
        axes[2, col].imshow(img)
        axes[2, col].imshow(attr_pred, cmap='inferno', alpha=0.55)
        axes[3, col].imshow(img)
        axes[3, col].imshow(attr_true, cmap='inferno', alpha=0.55)
        for row in range(4):
            axes[row, col].axis('off')
    labels = ['Input', '|IG| (pred class)', 'Overlay (pred class)', 'Overlay (true class)']
    for r, label in enumerate(labels):
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, ha='center', va='center', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
