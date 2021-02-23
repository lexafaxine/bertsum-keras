from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics
import numpy as np
from transformers import BertTokenizer, TFBertModel, BertConfig
import math


# Implement a position-wise-feedforward
class PositionwiseFeedForward(layers.Layer):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = layers.Dense(d_ff, input_shape=(d_model,), activation="gelu")
        self.w_2 = layers.Dense(d_model, input_shape=(d_ff,))
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)

    def call(self, x):
        inter = self.dropout_1(self.w_1(self.layer_norm(x)))
        output = self.dropout_2(self.w_2(inter))

        return output + x


# Implement a PositionalEncoding
class PositionalEncoding(layers.Layer):

    def __init__(self, dropout, dim, max_len=5000):
        pe = np.zeros([max_len, dim])
        position = np.arange(0, max_len, dtype=float)
        position = np.expand_dims(position, 1)
        div_term = np.exp(np.arange(0, dim, step=2, dtype=float) * -(math.log(10000.0))/dim)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, axis=0)

        super(PositionalEncoding, self).__init__()
        self.dropout = layers.Dropout(dropout)
        self.dim = dim
        self.pe = pe

    def call(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.shape[1]]

        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.shape[1]]


# Implement the encoder transformer layer
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = layers.MultiHeadAttention(num_heads=heads,
                                                   key_dim=d_model,
                                                   dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = tf.expand_dims(mask, axis=1)
        context = self.self_attn(input_norm, input_norm, input_norm, attention_mask=mask)
        out = self.dropout(context) + inputs

        return self.feed_forward(out)


# Implement the interval transformer encoder
class TransformerInterEncoder(layers.Layer):

    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layer=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layer
        self.pos_emb = PositionalEncoding(dropout, dim=d_model)
        self.transformer_inter = []
        for i in range(num_inter_layer):
            self.transformer_inter.append(TransformerEncoderLayer(d_model=d_model,
                                                                  heads=heads, d_ff=d_ff, dropout=dropout))
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.wo = layers.Dense(1, input_shape=(d_model,), use_bias=True, activation="sigmoid")

    def call(self, res, mask):
        batch_size, n_sents = res.shape[0], res.shape[1]
        pos_emb = self.pos_emb.pe[:, :n_sents]
        mask = tf.cast(mask, dtype=tf.float32)
        x = res * mask[:, :, None]
        x += pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)

        x = self.layer_norm(x)
        sent_scores = self.wo(x)
        sent_scores = tf.squeeze(sent_scores, axis=-1) * tf.cast(mask, dtype=tf.float32)

        return sent_scores

# Implement Bert and return the sent_vec
class Bert(layers.Layer):
    def __init__(self, bert_name, max_length):
        # self.output_dim = output_dim
        super(Bert, self).__init__()
        # Load transformers config
        if bert_name:
            self.config = BertConfig.from_pretrained(bert_name)
            self.config.output_hidden_states = True
        else:
            pass

        # Load bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_name, config=self.config)

        # Load the Transformers BERT model
        self.model = TFBertModel.from_pretrained(bert_name, config=self.config)

    def call(self, inputs, training):
        # def call(self, x, segs, mask):
        # return the the embedding of [CLS]
        outputs = self.model(inputs[0], token_type_ids=inputs[1], attention_mask=inputs[3], training=training)
        top_vec = outputs[0]
        cls_ids = inputs[2]
        index = inputs[-2]

        index = tf.expand_dims(index, axis=-1)
        index = tf.tile(index, [1, cls_ids.shape[1], 1])

        indices = tf.concat([index, cls_ids[:, :, None]], axis=-1)
        # tf.print(indices)
        res = tf.gather_nd(top_vec, indices)

        return res



# DEFINE a custom LOSS and METRIC
class CustomLossMetric(layers.Layer):

    def __init__(self, name=None):
        super(CustomLossMetric, self).__init__(name=name)
        self.loss_fn = losses.BinaryCrossentropy()

        self.accuracy_fn = metrics.BinaryAccuracy()

    def call(self, sent_scores, labels, mask_cls):
        labels = tf.dtypes.cast(labels, dtype=tf.float32)
        loss = self.loss_fn(sent_scores, labels)
        mask_cls = tf.dtypes.cast(mask_cls, dtype=tf.float32)
        # loss = (loss * mask_cls).sum()

        loss = tf.reduce_sum(loss * mask_cls)

        self.add_loss(loss)

        acc = self.accuracy_fn(sent_scores, labels)
        self.add_metric(acc, name="accuracy")

        return sent_scores


def create_model(bert_name, max_length, encoder, hidden_size, ff_size, heads, dropout, inter_layers):
    # get the encoder and bert layer
    if encoder == "classifier":
        pass
    if encoder == "baseline":
        pass
    if encoder == "rnn":
        pass
    if encoder == "transformer":
        encoder = TransformerInterEncoder(hidden_size, ff_size, heads, dropout, inter_layers)

    bert = Bert(bert_name, max_length=max_length)
    custom = CustomLossMetric(name="lossandmetrics")
    # Start of Keras Functional API #

    # define inputs
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
    cls_ids = tf.keras.layers.Input(shape=(100,), dtype=tf.int32)
    mask_cls = tf.keras.layers.Input(shape=(100,), dtype=tf.int32)
    helper = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)
    labels = tf.keras.layers.Input(shape=(100,), dtype=tf.int32)

    inputs = [input_ids, segment_ids, cls_ids, mask, mask_cls, helper, labels]

    # define graph and output
    res = bert(inputs, training=True)
    sent_scores = encoder(res, mask_cls)

    sent_scores = custom(sent_scores, labels, mask_cls)

    outputs = sent_scores

    # define opt and loss
    opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=5e-5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BertSum")
    model.compile(optimizer=opt)

    return model