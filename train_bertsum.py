from __future__ import print_function
import tensorflow as tf
import numpy as np
from data_loader import create_inputs
from model import create_model
import os

# flags = tf.flags
flags = tf.compat.v1.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "json_path", None,
    "The path of the json file"
)

flags.DEFINE_string(
    "bert_name", "bert-base-uncased",
    "The pretrained bert's name"
)

flags.DEFINE_integer(
    "min_src_ntokens", 3,
    "min words of a sentence"
)

flags.DEFINE_integer(
    "max_src_ntokens", 200,
    "max words of a sentence"
)

flags.DEFINE_integer(
    "max_nsents", 100,
    "max number of sentences"
)

flags.DEFINE_integer(
    "min_nsents", 3,
    "min number of sentences"
)

flags.DEFINE_string(
    "oracle_mode", None, "Oracle Mode"
)

flags.DEFINE_string(
    "mode", None, "train test or val"
)

flags.DEFINE_integer(
    "max_length", 512,
    "max sequence length of bert"
)

flags.DEFINE_bool(
    "is_test", False, "is test"
)

flags.DEFINE_string(
    "encoder", "transformer",
    "the encoder after bert, will be classifier, transformer, baseline or rnn"
)

flags.DEFINE_integer(
    "hidden_size", 768, "The hidden size of bert"
)

flags.DEFINE_integer(
    "ff_size", 512, "The ff size of attention in transformer"
)

flags.DEFINE_integer(
    "heads", 8, "Number of head of the multi head attention"
)

flags.DEFINE_float(
    "dropout", 0.1, "The dropout rate"
)

flags.DEFINE_integer(
    "inter_layers", 2, "Number of transformer layer"
)

def generate_train_batch(x_samples, batch_size):
    num_batches = len(x_samples[0]) // batch_size
    print(num_batches)
    if num_batches == 0:
        raise ValueError("ddd")
    # inputs =
    while True:
        batch_x = []
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            input_ids_batch = x_samples[0][start:end]
            labels_batch = x_samples[-1][start:end]
            segment_ids_batch = x_samples[1][start:end]
            cls_ids_batch = x_samples[3][start:end]
            mask_cls_batch = x_samples[4][start:end]
            mask_batch = x_samples[2][start:end]
            helper_batch = np.arange(0, batch_size, dtype=int)[:, None]

            x = [input_ids_batch, segment_ids_batch, cls_ids_batch, mask_batch, mask_cls_batch, helper_batch, labels_batch]
            # print("=================", batchIdx,"th batch====================")
            yield x, labels_batch


def main(_):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    json_path = FLAGS.json_path
    bert_name = FLAGS.bert_name
    min_src_ntokens = FLAGS.min_src_ntokens
    max_src_ntokens = FLAGS.max_src_ntokens
    max_nsents = FLAGS.max_nsents
    min_nsents = FLAGS.min_nsents
    oracle_mode = FLAGS.oracle_mode
    mode = FLAGS.mode
    max_length = FLAGS.max_length
    is_test = FLAGS.is_test
    encoder = FLAGS.encoder
    hidden_size = FLAGS.hidden_size
    ff_size = FLAGS.ff_size
    heads = FLAGS.heads
    dropout = FLAGS.dropout
    inter_layers = FLAGS.inter_layers

    inputs = create_inputs(json_path, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, oracle_mode,
                           mode, max_length, is_test)

    model = create_model(bert_name=bert_name, max_length=max_length, encoder=encoder, hidden_size=hidden_size,
                         ff_size=ff_size, heads=heads, dropout=dropout, inter_layers=inter_layers)

    model.summary()

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    train_generator = generate_train_batch(inputs, batch_size=8)

    num_train_steps = len(inputs[0]) // 8

    history = model.fit(x=train_generator, epochs=100,
                        verbose=1, validation_data=None,
                        validation_steps=None,
                        steps_per_epoch=num_train_steps,
                         callbacks=[cp_callback])

    return history

if __name__ == "__main__":
    flags.mark_flag_as_required("json_path")
    flags.mark_flag_as_required("oracle_mode")
    flags.mark_flag_as_required("mode")

    tf.compat.v1.app.run()