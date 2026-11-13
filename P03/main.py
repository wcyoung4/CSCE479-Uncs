import argparse
import importlib
import json
import os
import time
import numpy as np
import tensorflow as tf

import util
import model as model_lib

DATA_DIR = './tensorflow-datasets/'
results_path = './results/'
save_path = results_path + 'saved_models/'
figure_path = results_path + 'figures/'
epoch_int = 10
max_epoch = 50
patience = 5
batch_size = 64
seed = 479
c_labels = ['negative', 'positive']

# vectorization hyperparams
MAX_TOKENS = 10000
MAX_SEQ_LEN = 256

# architectures
architectures = {
    'lstm': {'type': 'lstm', 'filters': 100, 'kernel_size': 4, 'units': [256, 64]},
    'gru': {'type': 'gru', 'filters': 100, 'kernel_size': 4, 'units': [256, 64]},
    'transformer': {'type': 'transformer', 'filters': 100, 'kernel_size': 4, 'heads': 4, 'ff_dim': 200},
}

# learning rate, dropout
settings = [(1e-3, 0.2), (1e-3, 0.5), (1e-4, 0.2)]
l2_weight = 1e-4


def train_and_validate(seed, training_set, valid_set, vocab_size, max_epoch=max_epoch,
                       patience=patience, restart=False):
    """Train every architecture and setting with early stopping on validation loss.
    Save each completed run so an interrupted experiment can be restarted."""
    progress_path = results_path + 'training_progress.json'
    config = {
        'seed': seed,
        'max_epoch': max_epoch,
        'patience': patience,
        'batch_size': batch_size,
        'vocab_size': vocab_size,
        'max_seq_len': MAX_SEQ_LEN,
        'architectures': architectures,
        'settings': [list(setting) for setting in settings],
        'l2_weight': l2_weight,
    }

    if os.path.exists(progress_path) and not restart:
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        if progress['config'] != config:
            raise ValueError('Saved training settings differ from the current settings. '
                             'Use --restart to start a new experiment.')
        results = progress['results']
        histories = progress['histories']
        print('Found %d completed training runs.' % len(results))
    else:
        results = []
        histories = {}
        with open(progress_path, 'w') as f:
            json.dump({'config': config, 'results': results, 'histories': histories}, f, indent=2)

    completed = {row['run'] for row in results}

    for arch_name, arch in architectures.items():
        for lr, dropout in settings:
            name = util.run_name(arch_name, dropout, lr)
            if name in completed:
                if os.path.exists(save_path + name + '.keras'):
                    print('Skipping completed run: ' + name)
                    continue
                print('Saved model missing. Retraining: ' + name)
                results = [row for row in results if row['run'] != name]
                histories.pop(name, None)

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = model_lib.build_model(arch, dropout, lr, l2_weight, vocab_size, MAX_SEQ_LEN)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True)
            save_call = util.SaveModelEpoch(save_str=save_path + name + '_epoch', epoch_interval=epoch_int)

            start = time.perf_counter()
            record = model.fit(
                training_set,
                epochs=max_epoch,
                validation_data=valid_set,
                callbacks=[early_stop, save_call],
                verbose=2
            )
            elapsed = time.perf_counter() - start
            histories[name] = {key: [float(v) for v in values]
                               for key, values in record.history.items()}

            model.save(save_path + name + '.keras')
            train_loss, train_acc, train_auc = model.evaluate(training_set, verbose=0)
            val_loss, val_acc, val_auc = model.evaluate(valid_set, verbose=0)
            results.append({
                'run': name,
                'architecture': arch_name,
                'learning_rate': lr,
                'dropout': dropout,
                'l2': l2_weight,
                'params': int(model.count_params()),
                'epochs_run': len(record.history['loss']),
                'best_epoch': int(np.argmin(record.history['val_loss'])) + 1,
                'train_loss': float(train_loss),
                'train_accuracy': float(train_acc),
                'train_auc': float(train_auc),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc),
                'val_auc': float(val_auc),
                'train_seconds': float(elapsed),
            })

            with open(progress_path, 'w') as f:
                json.dump({'config': config, 'results': results, 'histories': histories}, f, indent=2)

            print(name + ': train_acc=%.4f  val_acc=%.4f  val_auc=%.4f  training=%.1f min'
                  % (train_acc, val_acc, val_auc, elapsed / 60))
            del model

    results.sort(key=lambda r: (-r['val_accuracy'], r['val_loss']))
    return results, histories


def main(seed, max_epoch, restart=False):
    importlib.invalidate_caches()
    importlib.reload(model_lib)
    importlib.reload(util)
    print('Using model definitions from:', model_lib.__file__)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    tf.keras.utils.set_random_seed(seed)
    training_set, valid_set, testing_set, vectorize_layer = util.load_data(
        DATA_DIR, MAX_TOKENS, MAX_SEQ_LEN, batch_size, seed)
    vocab = vectorize_layer.get_vocabulary()

    #architectures
    for arch_name, arch in architectures.items():
        tf.keras.backend.clear_session()
        model = model_lib.build_model(arch, settings[0][1], settings[0][0], l2_weight, len(vocab), MAX_SEQ_LEN)
        table = util.architecture_table(arch, model.count_params(), len(vocab), MAX_SEQ_LEN,
                                        settings, l2_weight, patience)
        util.plot_table(table, figure_path + 'architecture_' + arch_name + '.png',
                        title='%s architecture' % arch_name.upper())
        del model

    results, histories = train_and_validate(seed, training_set, valid_set, len(vocab),
                                            max_epoch=max_epoch, restart=restart)

    #results
    tf.keras.backend.clear_session()
    util.test_report(results, histories, testing_set, vocab, architectures, settings,
                     save_path, figure_path, results_path, c_labels, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--max_epoch', type=int, default=max_epoch)
    parser.add_argument('--restart', action='store_true', help='Discard saved progress and retrain all nine models')
    args = parser.parse_args()
    main(args.seed, args.max_epoch, args.restart)
