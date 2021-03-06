"""GloVe Embeddings + chars bi-LSTM + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

DATADIR = 'C:/Users/alien/Documents/PyCharm Projects/Address element extraction tf1/char lstm lstm crf'

# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def timenow():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':')


timeStart = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r', encoding='utf-8') as f_words, Path(tags).open('r', encoding='utf-8') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),  # (words, nwords)
               ([None, None], [None])),  # (chars, nchars)
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def graph_fn(features, labels, mode, params, reuse=None, getter=None):
    # Read vocabs and inputs
    num_tags = params['num_tags']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope('graph', reuse=reuse, custom_getter=getter):
        # Read vocabs and inputs
        dropout = params['dropout']
        (words, nwords), (chars, nchars) = features
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        vocab_words = tf.contrib.lookup.index_table_from_file(
            params['words'], num_oov_buckets=params['num_oov_buckets'])
        vocab_chars = tf.contrib.lookup.index_table_from_file(
            params['chars'], num_oov_buckets=params['num_oov_buckets'])
        with Path(params['chars']).open(encoding='utf-8') as f:
            num_chars = sum(1 for _ in f) + params['num_oov_buckets']

        # Char Embeddings
        char_ids = vocab_chars.lookup(chars)
        variable = tf.get_variable(
            'chars_embeddings', [num_chars, params['dim_chars']], tf.float32)
        char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
        char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                            training=training)

        # Char LSTM
        dim_words = tf.shape(char_embeddings)[1]
        dim_chars = tf.shape(char_embeddings)[2]
        flat = tf.reshape(char_embeddings, [-1, dim_chars, params['dim_chars']])
        t = tf.transpose(flat, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        _, (_, output_fw) = lstm_cell_fw(t, dtype=tf.float32,
                                         sequence_length=tf.reshape(nchars, [-1]))
        _, (_, output_bw) = lstm_cell_bw(t, dtype=tf.float32,
                                         sequence_length=tf.reshape(nchars, [-1]))
        output = tf.concat([output_fw, output_bw], axis=-1)
        char_embeddings = tf.reshape(output, [-1, dim_words, 50])

        # Word Embeddings
        word_ids = vocab_words.lookup(words)
        glove = np.load(params['glove'])['embeddings']  # np.array
        variable = np.vstack([glove, [[0.] * params['dim']]])
        variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
        word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

        # Concatenate Word and Char Embeddings
        embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
        embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

        # LSTM
        t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=dropout, training=training)

        # CRF
        logits = tf.layers.dense(output, num_tags)
        crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)

    return logits, crf_params


def ema_getter(ema):
    def _ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return _ema_getter


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    with Path(params['tags']).open(encoding='utf-8') as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
        params['num_tags'] = num_tags

    # Graph
    (words, nwords), (chars, nchars) = features
    logits, crf_params = graph_fn(features, labels, mode, params)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    # Moving Average
    variables = tf.get_collection('trainable_variables', 'graph')
    ema = tf.train.ExponentialMovingAverage(0.999)
    ema_op = ema.apply(variables)
    logits_ema, crf_params_ema = graph_fn(
        features, labels, mode, params, reuse=True, getter=ema_getter(ema))
    pred_ids_ema, _ = tf.contrib.crf.crf_decode(
        logits_ema, crf_params_ema, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        pred_strings_ema = reverse_vocab_tags.lookup(tf.to_int64(pred_ids_ema))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings,
            'pred_ids_ema': pred_ids_ema,
            'tags_ema': pred_strings_ema,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'acc_ema': tf.metrics.accuracy(tags, pred_ids_ema, weights),
            'pr': precision(tags, pred_ids, num_tags, indices, weights),
            'pr_ema': precision(tags, pred_ids_ema, num_tags, indices, weights),
            'rc': recall(tags, pred_ids, num_tags, indices, weights),
            'rc_ema': recall(tags, pred_ids_ema, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
            # 'f1_ema': f1(tags, pred_ids_ema, num_tags, indices, weights),
            'f1_ema': f1(tags, pred_ids_ema, num_tags, indices),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step(),
                var_list=variables)
            # train_op = tf.keras.optimizers.SGD(learning_rate=0.01, clipvalue=5.0).minimize(loss, var_list=variables)
            train_op = tf.group([train_op, ema_op])
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 300,
        'dim_chars': 100,
        'dropout': 0.125,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'char_lstm_size': 25,
        'lstm_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path('results/params.json').open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))


    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))


    def serving_input_receiver_fn():
        """Serving input_fn that builds features from placeholders
        Returns
        -------
        tf.estimator.export.ServingInputReceiver
        """
        words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
        nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
        chars = tf.placeholder(dtype=tf.string, shape=[None, None, None],
                               name='chars')
        nchars = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                name='nchars')
        receiver_tensors = {'words': words, 'nwords': nwords,
                            'chars': chars, 'nchars': nchars}
        features = {'words': words, 'nwords': nwords,
                    'chars': chars, 'nchars': nchars}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('val'), ftags('val'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=80)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator, 'f1_ema', 800, min_steps=8000, run_every_secs=80)  # f1_ema default, 500 default, min steps = 8000
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=80)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name, mode):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.{}.preds.txt'.format(name, mode)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds[mode]):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')


    for name in ['train', 'val']:
        for mode in ['tags', 'tags_ema']:
            write_predictions(name, mode)

print(timenow(), f'Training Completed. Time started: {timeStart}')
