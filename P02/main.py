import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import util
from model import build_model

DATA_DIR = '../tensorflow-datasets/'
results_path = './results/'
save_path = results_path + 'saved_models/'
figure_path = results_path + 'figures/'
arch_path = results_path + 'architectures/'
epoch_int = 25
max_epoch = 200
patience = 10
batch_size = 64
seed = 479
n_classes = 100

# two conv architectures, a small one and a deeper VGG style one with batch norm
architectures = {
    'small': {'filters': [32, 64], 'convs_per_block': 1, 'dense': [256], 'batch_norm': False},
    'deep': {'filters': [64, 128, 256], 'convs_per_block': 2, 'dense': [512], 'batch_norm': True},
}

# the 2x2x2 grid (L2 is on for every run, dropout is toggled)
learning_rates = [1e-3, 1e-4]
dropouts = [0.0, 0.3]
l2_weight = 5e-4
augment = True

os.makedirs(save_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)
os.makedirs(arch_path, exist_ok=True)

#partition data 90/10 train/validate
training_set, info = tfds.load('cifar100', split='train[:90%]', data_dir=DATA_DIR, with_info=True)
valid_set = tfds.load('cifar100', split='train[-10%:]', data_dir=DATA_DIR)
raw_test = tfds.load('cifar100', split='test', data_dir=DATA_DIR)

c_labels = info.features['label'].names
coarse_labels = info.features['coarse_label'].names

#set up data and batches
def prepare_data(data, shuffle=False):
    data = data.cache()
    if shuffle:
        data = data.shuffle(10000, seed=seed)

    # Convert integer class labels to one-hot vectors
    data = data.map(
        lambda ex: (ex['image'], tf.one_hot(ex['label'], depth=n_classes)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

training_set = prepare_data(training_set, shuffle=True)
valid_set = prepare_data(valid_set)
testing_set = prepare_data(raw_test)

#architectures
for arch_name, arch in architectures.items():
    table = util.architecture_table(arch, dropouts, l2_weight, augment)
    util.write_csv(table, arch_path + arch_name + '.csv')
    util.plot_architecture_table(table, figure_path + 'architecture_' + arch_name + '.png',
                                 title='%s CNN architecture' % arch_name.title())
    print('\n' + arch_name + ' architecture:\n')
    util.print_table(table)


results = []
histories = {}

for arch_name, arch in architectures.items():
    for lr in learning_rates:
        for dropout in dropouts:
            name = util.run_name(arch_name, dropout, lr)
            tf.keras.utils.set_random_seed(seed)
            model = build_model(arch, dropout, lr, l2_weight, augment)

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            save_call = util.SaveModelEpoch(save_str=save_path + name + '_epoch', epoch_interval=epoch_int)

            record = model.fit(
                    training_set,
                    epochs=max_epoch,
                    batch_size=batch_size,
                    validation_data=valid_set,
                    callbacks=[early_stop, save_call],
                    verbose=2
                )
            histories[name] = record.history

            model.save(save_path + name + '.keras')

            train_loss, train_acc, train_top5 = model.evaluate(training_set, verbose=0)
            val_loss, val_acc, val_top5 = model.evaluate(valid_set, verbose=0)
            results.append({
                'run': name,
                'architecture': arch_name,
                'learning_rate': lr,
                'dropout': dropout,
                'l2': l2_weight,
                'params': model.count_params(),
                'epochs_run': len(record.history['loss']),
                'best_epoch': int(np.argmin(record.history['val_loss'])) + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_top5_accuracy': train_top5,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_top5_accuracy': val_top5,
            })
            print(name + ': train_acc=%.4f  val_acc=%.4f  val_top5=%.4f' % (train_acc, val_acc, val_top5))

#results
results.sort(key=lambda r: r['val_accuracy'], reverse=True)
with open(results_path + 'histories.json', 'w') as f:
    json.dump(histories, f, indent=2)

print('\n\nValidation results (ranked by validation accuracy):\n')
util.print_table(results)

best = results[0]
print('\nSelected on validation accuracy: ' + best['run'])

# raw test arrays in the same order as testing_set
raw_batches = list(tfds.as_numpy(raw_test.batch(1000)))
images = np.concatenate([b['image'] for b in raw_batches], axis=0)
true_labels = np.concatenate([b['label'] for b in raw_batches], axis=0)
true_coarse = np.concatenate([b['coarse_label'] for b in raw_batches], axis=0)

# Evaluate the held-out test set after model selection is fixed
correctness = {}
for row in results:
    test_model = tf.keras.models.load_model(save_path + row['run'] + '.keras')
    test_loss, test_acc, test_top5 = test_model.evaluate(testing_set, verbose=0)
    test_probs = test_model.predict(testing_set, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    top5_preds = np.argsort(-test_probs, axis=1)[:, :5]
    correctness[row['run']] = {
        'test_accuracy': test_preds == true_labels,
        'test_top5_accuracy': np.any(top5_preds == true_labels[:, None], axis=1),
    }
    row['test_loss'] = test_loss
    row['test_accuracy'] = test_acc
    row['test_top5_accuracy'] = test_top5
    if row['run'] == best['run']:
        probs = test_probs
        preds = test_preds
    print(row['run'] + ': test_acc=%.4f  test_top5=%.4f' % (test_acc, test_top5))

util.write_csv(results, results_path + 'results.csv')
print('\n\nAll runs (ranked by validation accuracy):\n')
util.print_table(results)

model = tf.keras.models.load_model(save_path + best['run'] + '.keras')
n_test = len(true_labels)
test_acc = np.mean(preds == true_labels)
test_top5 = util.top_k_accuracy(probs, true_labels, k=5)

# each fine class belongs to one superclass
fine_to_coarse = np.zeros(n_classes, dtype=int)
fine_to_coarse[true_labels] = true_coarse
coarse_preds = fine_to_coarse[preds]
coarse_acc = np.mean(coarse_preds == true_coarse)

print('\nConfidence intervals on the best model (95%%, n = %d):' % n_test)
for label, acc in [('Top-1 accuracy', test_acc), ('Top-5 accuracy', test_top5), ('Superclass accuracy', coarse_acc)]:
    half, low, high = util.accuracy_ci(acc, n_test)
    print('%-20s %.4f +/- %.4f  (CI [%.4f, %.4f])' % (label, acc, half, low, high))
    half, low, high = util.accuracy_ci(1 - acc, n_test)
    print('%-20s %.4f +/- %.4f  (CI [%.4f, %.4f])' % ('  error', 1 - acc, half, low, high))

comparisons = []
for arch_name in architectures:
    for lr in learning_rates:
        with_do = util.find_run(results, util.run_name(arch_name, max(dropouts), lr))
        without_do = util.find_run(results, util.run_name(arch_name, 0.0, lr))
        comparisons.append(util.error_difference_ci('dropout %s vs none' % max(dropouts), with_do, without_do, correctness))
    fast = util.find_run(results, util.run_name(arch_name, max(dropouts), learning_rates[0]))
    slow = util.find_run(results, util.run_name(arch_name, max(dropouts), learning_rates[1]))
    comparisons.append(util.error_difference_ci('lr %.0e vs %.0e' % (learning_rates[0], learning_rates[1]), fast, slow, correctness))
for lr in learning_rates:
    for dropout in dropouts:
        deep = util.find_run(results, util.run_name('deep', dropout, lr))
        small = util.find_run(results, util.run_name('small', dropout, lr))
        comparisons.append(util.error_difference_ci('deep vs small', deep, small, correctness))
comparisons.append(util.error_difference_ci('best vs runner-up', results[0], results[1], correctness))
comparisons.append(util.error_difference_ci('best vs runner-up', results[0], results[1], correctness, key='test_top5_accuracy'))

util.write_csv(comparisons, results_path + 'comparisons.csv')
print('\n95% CIs on the difference in test error (negative means model A is better,')
print('significant is True when the interval does not contain 0):\n')
util.print_table(comparisons)

# FIGURES
cm = util.confusion_matrix(true_labels, preds, n_classes)
util.plot_confusion_matrix(cm, figure_path + 'confusion_matrix.png', c_labels, annotate=False,
                           title='Confusion matrix, 100 classes')
cm_coarse = util.confusion_matrix(true_coarse, coarse_preds, len(coarse_labels))
util.plot_confusion_matrix(cm_coarse, figure_path + 'confusion_matrix_superclass.png', coarse_labels,
                           title='Confusion matrix, 20 superclasses')
util.plot_histories(histories, figure_path + 'training_curves.png')

#random images
rng = np.random.default_rng(seed)
sample = rng.choice(n_test, size=24, replace=False)
errors = util.most_confident_errors(probs, preds, true_labels, n=24)
util.plot_prediction_grid(images, true_labels, preds, figure_path + 'predictions.png', c_labels,
                          indices=sample, title='Random test predictions (' + best['run'] + ')')
util.plot_prediction_grid(images, true_labels, preds, figure_path + 'misclassified.png', c_labels,
                          indices=errors, title='Most confident misclassifications')

#per class accuracy
per_class = []
for c in range(n_classes):
    others = np.where(np.arange(n_classes) == c, -1, cm[c])
    per_class.append({
        'class': c_labels[c],
        'superclass': coarse_labels[fine_to_coarse[c]],
        'n': int(cm[c].sum()),
        'accuracy': cm[c, c] / cm[c].sum(),
        'most_confused_with': c_labels[int(np.argmax(others))],
        'times_confused': int(np.max(others)),
    })
util.write_csv(per_class, results_path + 'per_class.csv')
ranked = sorted(per_class, key=lambda r: r['accuracy'], reverse=True)
print('\nTen easiest classes:\n')
util.print_table(ranked[:10])
print('\nTen hardest classes:\n')
util.print_table(ranked[-10:])

#integrated gradients take the best, the median and the worst class by test accuracy. Within each class compare the most confident correct, the least confident correct and the most confident wrong image
order = np.argsort([-r['accuracy'] for r in per_class], kind='stable')
ig_classes = [('easiest', order[0]), ('median', order[n_classes // 2]), ('hardest', order[-1])]

ig_rows = []
for tag, cls in ig_classes:
    entries = []
    for criterion, idx in util.select_ig_examples(probs, preds, true_labels, cls):
        ig_pred, delta_pred = util.integrated_gradients(model, images[idx], int(preds[idx]))
        ig_true, delta_true = util.integrated_gradients(model, images[idx], int(true_labels[idx]))
        entries.append({'index': idx, 'criterion': criterion, 'true': int(true_labels[idx]),
                        'pred': int(preds[idx]), 'confidence': float(probs[idx, preds[idx]]),
                        'ig_pred': ig_pred, 'ig_true': ig_true})

        mass = np.abs(ig_pred).sum(axis=-1)
        ig_rows.append({
            'class_rank': tag,
            'class': c_labels[cls],
            'criterion': criterion,
            'test_index': int(idx),
            'predicted': c_labels[preds[idx]],
            'confidence': float(probs[idx, preds[idx]]),
            'true_class_prob': float(probs[idx, true_labels[idx]]),
            'ig_sum_pred': float(ig_pred.sum()),
            'completeness_gap_pred': float(ig_pred.sum()) - delta_pred,
            'ig_sum_true': float(ig_true.sum()),
            'completeness_gap_true': float(ig_true.sum()) - delta_true,
            'centre_mass': float(mass[8:24, 8:24].sum() / max(mass.sum(), 1e-12)),
        })
    if entries:
        util.plot_ig_grid(images, entries, figure_path + 'ig_' + tag + '_' + c_labels[cls] + '.png', c_labels,
                          title='Integrated Gradients, %s class (%s, acc %.0f%%), black baseline'
                                % (tag, c_labels[cls], 100 * per_class[cls]['accuracy']))

entries = []
for idx in errors[:6]:
    ig_pred, _ = util.integrated_gradients(model, images[idx], int(preds[idx]))
    ig_true, _ = util.integrated_gradients(model, images[idx], int(true_labels[idx]))
    entries.append({'index': idx, 'criterion': 'confident error', 'true': int(true_labels[idx]),
                    'pred': int(preds[idx]), 'confidence': float(probs[idx, preds[idx]]),
                    'ig_pred': ig_pred, 'ig_true': ig_true})
util.plot_ig_grid(images, entries, figure_path + 'ig_confident_errors.png', c_labels,
                  title='Integrated Gradients on the most confident misclassifications')

util.write_csv(ig_rows, results_path + 'ig_examples.csv')
print('\nIntegrated Gradients examples (completeness gap should be close to 0):\n')
util.print_table(ig_rows)
