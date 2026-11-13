import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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

#data
def load_data(data_dir, max_tokens, max_seq_len, batch_size, seed):
    #partition data 90/10 train/validate
    training_set = tfds.load('imdb_reviews', split='train[:90%]', data_dir=data_dir)
    valid_set = tfds.load('imdb_reviews', split='train[-10%:]', data_dir=data_dir)
    testing_set = tfds.load('imdb_reviews', split='test', data_dir=data_dir)

    # Create TextVectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_seq_len)
    train_text = training_set.map(lambda x: x['text'])
    vectorize_layer.adapt(train_text.batch(1024))

    #set up data and batches
    def prepare_data(data, shuffle=False):
        data = data.cache()
        if shuffle:
            data = data.shuffle(10000, seed=seed)

        data = data.batch(batch_size)
        # binary cross entropy prep
        data = data.map(
            lambda ex: (vectorize_layer(ex['text']), tf.cast(ex['label'], tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        data = data.prefetch(tf.data.AUTOTUNE)
        return data

    return prepare_data(training_set, shuffle=True), prepare_data(valid_set), prepare_data(testing_set), vectorize_layer

def run_name(arch_name, dropout_rate, lr):
    return '%s_lr%.0e_do%s' % (arch_name, lr, dropout_rate)

def find_run(results, name):
    for row in results:
        if row['run'] == name:
            return row

def accuracy_ci(accuracy, n):
    half = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n)
    return half, max(0.0, accuracy - half), min(1.0, accuracy + half)

def error_difference_ci(label, row_a, row_b, correctness, key='test_accuracy'):
    correct_a = np.asarray(correctness[row_a['run']][key], dtype=bool)
    correct_b = np.asarray(correctness[row_b['run']][key], dtype=bool)
    if len(correct_a) != len(correct_b):
        raise ValueError('Models must use the same test examples')

    err_a = (~correct_a).astype(float)
    err_b = (~correct_b).astype(float)
    per_review_diff = err_a - err_b
    diff = float(np.mean(per_review_diff))
    half = float(1.96 * np.std(per_review_diff, ddof=1) / np.sqrt(len(per_review_diff)))
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

def confusion_matrix(trues, preds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    return cm

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
def architecture_table(arch, params, vocab_size, max_seq_len, settings, l2_weight, patience):
    embedding_size = int(np.sqrt(vocab_size))
    rows = [
        {'stage': 'Input', 'configuration': 'Raw review text, lowercased with punctuation stripped, split on whitespace'},
        {'stage': 'Vectorization', 'configuration': 'TextVectorization, %d token vocabulary adapted on the training split only\n'
                                                    'padded or truncated to %d tokens (0 = padding, 1 = unknown word)' % (vocab_size, max_seq_len)},
        {'stage': 'Embedding', 'configuration': 'Embedding(%d, %d) -> Dropout(d), padding mask = (token != 0)' % (vocab_size, embedding_size)},
        {'stage': 'Query / value', 'configuration': 'Conv1D(%d, kernel %d, stride 1, same) for the query and a second one for the value'
                                                    % (arch['filters'], arch['kernel_size'])},
    ]

    if arch['type'] == 'transformer':
        heads = arch['heads']
        rows.append({'stage': 'Attention', 'configuration': 'MultiHeadAttention(%d heads, key_dim %d, dropout d), padding masked\n'
                                                            '-> Dropout(d) -> Add(query) -> LayerNorm' % (heads, arch['filters'] // heads)})
        rows.append({'stage': 'Feed forward', 'configuration': 'Dense(%d, ReLU) -> Dense(%d) -> Dropout(d) -> Add -> LayerNorm\n'
                                                               '-> masked GlobalAveragePooling1D' % (arch['ff_dim'], arch['filters'])})
    else:
        rows.append({'stage': 'Attention', 'configuration': 'MultiHeadAttention(1 head, key_dim %d, dropout d), padding masked\n'
                                                            '-> Concatenate([query, attention]), %d x %d' % (arch['filters'], max_seq_len, 2 * arch['filters'])})
        cell = arch['type'].upper()
        layers = ' -> '.join('%s(%d%s)' % (cell, u, ', return_sequences' if i < len(arch['units']) - 1 else '')
                             for i, u in enumerate(arch['units']))
        rows.append({'stage': 'Recurrent', 'configuration': layers + '\ntanh outputs, sigmoid gates, masked so padding is skipped'})

    rows.append({'stage': 'Output', 'configuration': 'Dropout(d) -> Dense(1, Sigmoid), P(positive review)'})
    rows.append({'stage': 'Regularization',
                 'configuration': 'L2 = %.0e on Conv1D, RNN input kernels and FFN Dense kernels, dropout d in {%s}'
                                  % (l2_weight, ', '.join(str(d) for d in sorted(set(d for _, d in settings))))})
    rows.append({'stage': 'Training',
                 'configuration': 'Adam (clipnorm 1.0), binary cross-entropy, (lr, d) in {%s}\nearly stopping on validation loss, patience %d'
                                  % (', '.join('(%.0e, %s)' % s for s in settings), patience)})
    rows.append({'stage': 'Parameters', 'configuration': '{:,} trainable'.format(params)})
    return rows

def plot_table(rows, path, title):
    columns = list(rows[0].keys())
    cells = [['%.4f' % row[c] if isinstance(row[c], float) else str(row[c]) for c in columns] for row in rows]
    lengths = [3 + max(len(line) for text in [c] + [cell[i] for cell in cells] for line in text.split('\n'))
               for i, c in enumerate(columns)]
    heights = [0.36 + 0.27 * max(text.count('\n') for text in cell) for cell in cells]
    fig_height = sum(heights) + 0.37 + 0.9
    bottom = 0.2 / fig_height
    span = 1 - 0.7 / fig_height - bottom
    fig, ax = plt.subplots(figsize=(min(max(0.1 * sum(lengths) + 1, 7), 28), fig_height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=[c[0].upper() + c[1:].replace('_', ' ') for c in columns],
                     loc='center', cellLoc='left', colWidths=[l / sum(lengths) for l in lengths],
                     bbox=[0.0, bottom, 1.0, span])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#e6eef7')
            cell.set_text_props(weight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f5f7fa')
        if c == 0 and r != 0:
            cell.set_text_props(weight='bold')
        cell.PAD = 0.025
    total = sum(heights) + 0.37
    for i, height in enumerate([0.37] + heights):
        for j in range(len(columns)):
            table[i, j].set_height(height / total * span)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1 - 0.3 / fig_height)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.16)
    plt.close(fig)

#FIGURES
def plot_confusion_matrix(cm, path, class_names, normalize=True, annotate=True, title='Confusion matrix'):
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    size = 4.5 if n <= 2 else 8.5
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=1 if normalize else data.max())
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title + (' (row-normalized)' if normalize else ''), fontsize=9)
    if annotate:
        threshold = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if normalize:
                    text = '{:.1%}\n(n = {})'.format(data[i, j], int(cm[i, j]))
                else:
                    text = str(int(cm[i, j]))
                ax.text(j, i, text, ha='center', va='center', fontsize=10,
                        color='white' if data[i, j] > threshold else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_histories(histories, path):
    """Training / validation curves for every run in the grid."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    cmap = plt.get_cmap('tab10')
    panels = [('loss', 'Loss', 'Cross-entropy + L2'),
              ('accuracy', 'Accuracy', 'Accuracy'),
              ('auc', 'ROC AUC', 'AUC')]
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

def most_confident_errors(probs, preds, trues, n=24):
    """Indices of the misclassifications the model was most confident about."""
    wrong = np.flatnonzero(preds != trues)
    if len(wrong) == 0:
        return wrong
    confidence = probs[wrong, preds[wrong]]
    return wrong[np.argsort(-confidence)][:n]

#attention
def attention_scores(model, tokens):
    """Attention map (query word x value word) for a batch of token ids, averaged over the heads for the transformer."""
    query_value = tf.keras.Model(model.inputs, [model.get_layer('query').output, model.get_layer('value').output])
    tokens = tf.constant(tokens)
    query, value = query_value(tokens, training=False)
    mask = tf.not_equal(tokens, 0)
    layer = model.get_layer('attention')
    attention_mask = tf.logical_and(mask[:, :, None], mask[:, None, :])
    _, scores = layer(query, value, attention_mask=attention_mask, return_attention_scores=True)
    scores = tf.reduce_mean(scores, axis=1)
    return scores.numpy()

def plot_attention(attention_maps, token_labels, titles, path, suptitle):
    """Side by side attention heatmaps for one review."""
    fig, ax = plt.subplots(nrows=1, ncols=len(attention_maps), figsize=(6 * len(attention_maps), 5.5), squeeze=False)
    ax = ax[0]
    max_attention = max(np.max(a) for a in attention_maps)

    for c in range(len(attention_maps)):
        im = ax[c].imshow(attention_maps[c], cmap='plasma', vmin=0, vmax=max_attention)
        ax[c].set_xticks(range(len(token_labels)))
        ax[c].set_xticklabels(token_labels, rotation=90, fontsize=7)
        ax[c].set_yticks(range(len(token_labels)))
        ax[c].set_yticklabels(token_labels, fontsize=7)
        ax[c].set_title(titles[c], fontsize=9)
    fig.colorbar(im, ax=ax[-1], fraction=0.046)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

#test set
def test_report(results, histories, testing_set, vocab, architectures, settings, save_path, figure_path, results_path, c_labels, seed):
    n_classes = len(c_labels)
    with open(results_path + 'histories.json', 'w') as f:
        json.dump(histories, f, indent=2)

    best = results[0]
    print('\nSelected on validation accuracy: ' + best['run'])

    # raw test arrays
    tokens = np.concatenate([x.numpy() for x, y in testing_set], axis=0)
    true_labels = np.concatenate([y.numpy() for x, y in testing_set], axis=0).astype(int)
    n_test = len(true_labels)

    # Evaluate withheld test set
    correctness = {}
    run_preds = {}
    for row in results:
        test_model = tf.keras.models.load_model(save_path + row['run'] + '.keras')
        test_loss, _, test_auc = test_model.evaluate(testing_set, verbose=0)
        p_pos = test_model.predict(testing_set, verbose=0).ravel()
        test_probs = np.stack([1 - p_pos, p_pos], axis=1)
        test_preds = np.argmax(test_probs, axis=1)
        correctness[row['run']] = {'test_accuracy': test_preds == true_labels}
        run_preds[row['run']] = test_preds
        test_acc = float(np.mean(test_preds == true_labels))
        half, low, high = accuracy_ci(test_acc, n_test)
        row['test_loss'] = test_loss
        row['test_accuracy'] = test_acc
        row['test_ci_low'] = low
        row['test_ci_high'] = high
        row['test_auc'] = float(test_auc)
        if row['run'] == best['run']:
            probs = test_probs
            preds = test_preds

    summary = []
    for row in results:
        summary.append({'run': row['run'], 'params': '{:,}'.format(row['params']), 'epochs_run': row['epochs_run'],
                        'best_epoch': row['best_epoch'], 'train_accuracy': row['train_accuracy'], 'val_accuracy': row['val_accuracy'],
                        'test_accuracy': row['test_accuracy'], '95% CI': '[%.4f, %.4f]' % (row['test_ci_low'], row['test_ci_high']),
                        'test_auc': row['test_auc']})
    plot_table(summary, figure_path + 'results_table.png', 'All runs, ranked by validation accuracy')
    print('\n\nAll runs (ranked by validation accuracy):\n')
    print_table(results)

    half, _, _ = accuracy_ci(best['test_accuracy'], n_test)
    print('\nBest model test accuracy: %.4f +/- %.4f (95%% CI [%.4f, %.4f], n = %d)'
          % (best['test_accuracy'], half, best['test_ci_low'], best['test_ci_high'], n_test))

    base_lr, base_do = settings[0]
    comparisons = []
    for arch_name in architectures:
        base = find_run(results, run_name(arch_name, base_do, base_lr))
        for lr, dropout in settings[1:]:
            other = find_run(results, run_name(arch_name, dropout, lr))
            if dropout != base_do:
                label = 'dropout %s vs %s' % (dropout, base_do)
            else:
                label = 'lr %.0e vs %.0e' % (lr, base_lr)
            comparisons.append(error_difference_ci(label, other, base, correctness))
    arch_names = list(architectures)
    for lr, dropout in settings:
        for i in range(len(arch_names)):
            for j in range(i + 1, len(arch_names)):
                row_a = find_run(results, run_name(arch_names[i], dropout, lr))
                row_b = find_run(results, run_name(arch_names[j], dropout, lr))
                comparisons.append(error_difference_ci('%s vs %s' % (arch_names[i], arch_names[j]), row_a, row_b, correctness))
    comparisons.append(error_difference_ci('best vs runner-up', results[0], results[1], correctness))

    comparison_rows = [{'comparison': c['comparison'], 'model_a': c['model_a'], 'model_b': c['model_b'], 'error_a': c['error_a'],
                 'error_b': c['error_b'], 'difference': c['difference'], '95% CI': '[%.4f, %.4f]' % (c['ci_low'], c['ci_high']),
                 'significant': c['significant']} for c in comparisons]
    plot_table(comparison_rows, figure_path + 'comparisons_table.png',
               'Difference in test error, negative means model A is better')
    print('\n95% CIs on the difference in test error (negative means model A is better,')
    print('significant is True when the interval does not contain 0):\n')
    print_table(comparisons)

    # FIGURES
    for row in results:
        cm_run = confusion_matrix(true_labels, run_preds[row['run']], n_classes)
        plot_confusion_matrix(cm_run, figure_path + 'confusion_matrix_' + row['run'] + '.png', c_labels,
                              title='Confusion matrix, ' + row['run'])
    cm = confusion_matrix(true_labels, preds, n_classes)
    plot_confusion_matrix(cm, figure_path + 'confusion_matrix.png', c_labels,
                          title='Confusion matrix, best model (' + best['run'] + ')')
    plot_histories(histories, figure_path + 'training_curves.png')

    #per class accuracy
    per_class = []
    for c in range(n_classes):
        acc = cm[c, c] / cm[c].sum()
        half, low, high = accuracy_ci(acc, int(cm[c].sum()))
        per_class.append({
            'class': c_labels[c],
            'n': int(cm[c].sum()),
            'accuracy': acc,
            'ci_low': low,
            'ci_high': high,
        })
    plot_table(per_class, figure_path + 'per_class_table.png', 'Per-class test accuracy, best model (' + best['run'] + ')')
    print('\nPer-class test accuracy (best model):\n')
    print_table(per_class)

    errors = most_confident_errors(probs, preds, true_labels, n=24)
    error_rows = []
    for idx in errors:
        words = [vocab[t] for t in tokens[idx] if t != 0]
        error_rows.append({'test_index': int(idx), 'true': c_labels[true_labels[idx]], 'predicted': c_labels[preds[idx]],
                           'confidence': float(probs[idx, preds[idx]]), 'tokens': len(words),
                           'start_of_review': textwrap.fill(' '.join(words[:30]), 70)})
    if error_rows:
        plot_table(error_rows[:12], figure_path + 'misclassified_table.png',
                   'Most confident misclassifications, best model (' + best['run'] + ')')

    with open(results_path + 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    #attention maps
    arch_best = {}
    for row in results:
        arch_best.setdefault(row['architecture'], row)
    arch_models = {a: tf.keras.models.load_model(save_path + r['run'] + '.keras') for a, r in arch_best.items()}

    rng = np.random.default_rng(seed)
    sample = rng.choice(n_test, size=3, replace=False)
    n_show = 30
    for tag, idx in [('random', i) for i in sample] + [('error', i) for i in errors[:2]]:
        n = min(n_show, int(np.sum(tokens[idx] != 0)))
        token_labels = [vocab[t] for t in tokens[idx][:n]]
        attention_maps = []
        titles = []
        for arch_name, model in arch_models.items():
            attention_maps.append(attention_scores(model, tokens[idx:idx + 1])[0, :n, :n])
            p = float(model.predict(tokens[idx:idx + 1], verbose=0)[0, 0])
            titles.append('%s, P(positive) = %.2f' % (arch_best[arch_name]['run'], p))
        plot_attention(attention_maps, token_labels, titles, figure_path + 'attention_%s_%d.png' % (tag, idx),
                       'Test review %d (true: %s), first %d tokens' % (idx, c_labels[true_labels[idx]], n))
