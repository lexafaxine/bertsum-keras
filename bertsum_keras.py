from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os, re, logging
from transformers import BertTokenizer, TFBertModel, BertConfig
import json
import math

NUM_TRAIN_EPOCHS = 100
BATCH_SIZE = 16
HIDDEN_SIZE = 30


# =============================Preprocess============================== #

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class BertData(object):
    def __init__(self, input_ids, labels, segments_ids, cls_ids, mask_cls, mask, src_txt, tgt_txt):
        self.input_ids = input_ids
        self.labels = labels
        self.segments_ids = segments_ids
        self.cls_ids = cls_ids
        self.mask_cls = mask_cls
        self.mask_cls = mask_cls
        self.mask = mask
        self.src_txt = src_txt
        self.tgt_txt = tgt_txt


# Preprocess the data for the input of BERT(cnndm version)

class Processor(object):
    def __init__(self, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents):

        self.min_src_ntokens = min_src_ntokens
        self.max_src_ntokens = max_src_ntokens
        self.max_nsents = max_nsents
        self.min_nsents = min_nsents

        if bert_name:
            config = BertConfig.from_pretrained(bert_name)
            config.output_hidden_states = False
        else:
            pass

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_name, config=config)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, tgt, oracle_ids):
        # oracle_ids is the extract summarization of the text
        if (len(src) == 0):
            return None

        # =========Filter the src data=========#

        original_src_txt = [' '.join(s) for s in src]
        labels = [0] * len(src)

        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens)]
        src = [src[i][:self.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]

        assert len(src) == len(labels)
        src = src[:self.max_nsents]
        labels = labels[:self.max_nsents]

        if (len(src) < self.min_nsents):
            return None

        # ==============Start Tokenize=============#
        # [jdidjids dsjds jdsijdsi jdsidjsi, djskdjs jdskjdsk djskdjsk, ...]
        # xxx [SEP][CLS] dsdsklds[SEP][CLS]dkdkslds[SEP][CLS]...
        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP][CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]

        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []

        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        # print("cls_vid=", self.cls_vid)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        # print("*****cls_ids******=", cls_ids)
        labels = labels[:len(cls_ids)]
        assert len(cls_ids) == len(labels)

        if sum(labels) == 0:
            # all elements in labels is 0: dont make sense
            return None

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # ======================PAD=========================== #

        mask_cls = [0 for _ in range(100)]
        mask = [0 for _ in range(512)]

        padding_length_subtoken = 512 - len(src_subtoken_idxs)
        if padding_length_subtoken > 0:
            src_subtoken_idxs = src_subtoken_idxs + ([self.pad_vid] * padding_length_subtoken)
            segments_ids = segments_ids + ([0] * padding_length_subtoken)

        for i in range(0, 512):
            if src_subtoken_idxs[i] != self.pad_vid:
                mask[i] = 1
            else:
                mask[i] = 0

        padding_length_sent = self.max_nsents - len(labels)
        if padding_length_sent > 0:
            labels = labels + ([0] * padding_length_sent)
            cls_ids = cls_ids + ([-1] * padding_length_sent)
        #
        # mask_cls = (cls_ids != -1)
        assert len(cls_ids) == len(labels) == len(mask_cls)
        for i in range(0, 100):
            if cls_ids[i] == -1:
                mask_cls[i] = 0
            else:
                mask_cls[i] = 1

        # cls_ids[cls_ids == -1] = 0
        for i in range(0, 100):
            if cls_ids[i] == -1:
                cls_ids[i] = 0

        cnn_data = BertData(input_ids=src_subtoken_idxs, labels = labels, segments_ids=segments_ids, cls_ids=cls_ids,
                            mask_cls=mask_cls, mask=mask, src_txt=src_txt,
                            tgt_txt=tgt_txt)
        return cnn_data


def create_train_examples(json_path, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, oracle_mode):
    examples = []
    print("Preprocessing %s" % json_path)
    count = 0

    bert_data = Processor(bert_name=bert_name,
                        min_src_ntokens=min_src_ntokens,
                        max_src_ntokens=max_src_ntokens,
                        max_nsents=max_nsents,
                        min_nsents=min_nsents)

    for i in range(0, 1):

        json_file = os.path.join(json_path, ".train." + str(i) + ".json")

        print("preprocessing "+json_file)

        raw_datas = json.load(open(json_file))

        for data in raw_datas:
            source, target = data['src'], data['tgt']
            if (oracle_mode == "greedy"):
                oracle_ids = greedy_selection(source, target, 3)
            else:
                pass

            example = bert_data.preprocess(src=source, tgt=target, oracle_ids=oracle_ids)
            if (example is None):
                continue

            else:
                count += 1
                examples.append(example)

                if count == 100:
                    break

    print("Finish preprocessing")
    logging.info("Finish preprocessing")

    return examples


# ===========Create Input=====================#

# 将examples中的bert_data对象转换为array

def create_inputs(examples, is_test=False):
    dataset_dict = {
        "input_ids": [],
        "labels": [],
        "segments_ids": [],
        "cls_ids": [],
        "mask_cls": [],
        "mask": [],
        "src_txt": [],
        "tgt_txt": []
    }

    for example in examples:
        for key in dataset_dict:
            dataset_dict[key].append(getattr(example, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    if is_test:
        inputs = [
            dataset_dict["input_ids"],
            dataset_dict["labels"],
            dataset_dict["segments_ids"],
            dataset_dict["cls_ids"],
            dataset_dict["mask_cls"],
            dataset_dict["mask"],
            dataset_dict["src_txt"],
            dataset_dict["tgt_txt"],
        ]
    else:
        inputs = [
            dataset_dict["input_ids"],
            dataset_dict["labels"],
            dataset_dict["segments_ids"],
            dataset_dict["cls_ids"],
            dataset_dict["mask_cls"],
            dataset_dict["mask"]
        ]

    print("inputs shape=",(len(inputs), len(inputs[0])))
    return inputs


# ========================================DEFINE the model===================================#


# Implemnt a position-wise-feedforward
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
        # pe = tf.zeros([max_len, dim])
        pe = np.zeros([max_len, dim])
        # position = tf.range(0, max_len, dtype=tf.float32)
        # position = tf.expand_dims(position, axis=1)
        position = np.arange(0, max_len, dtype=float)
        position = np.expand_dims(position, 1)
        # div_term = tf.exp(tf.range(0, dim, 2, dtype=tf.float32) *
        #                   -(tf.math.log(10000.0) / dim))
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


# Implement a RNN encoder
class RNNencoder(tf.keras.Model):
    pass


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
        outputs = self.model(inputs[0], token_type_ids=inputs[1], attention_mask=inputs[-3], training=training)
        top_vec = outputs[0]
        cls_ids = inputs[2]
        index = inputs[-1]

        index = tf.expand_dims(index, axis=-1)

        index = tf.tile(index, [1, cls_ids.shape[1], 1])

        indices = tf.concat([index, cls_ids[:, :, None]], axis=-1)

        tf.print(indices)

        res = tf.gather_nd(top_vec, indices)

        return res


class Summarizer(object):

    def __init__(self, bert_name, max_length, encoder, hidden_size, ff_size, heads, dropout, inter_layers):
        # Load the bert layer

        self.bert_name = bert_name
        self.max_length = max_length
        self.bert = Bert(bert_name=self.bert_name, max_length=self.max_length)
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.heads = heads
        self.dropout = dropout
        self.inter_layers = inter_layers

        if (encoder == 'classifier'):
            model = tf.keras.Sequential()
            model.add(layers.Dense(1, input_shape=(self.hidden_size,), activation='sigmoid'))
            self.encoder = model
            # self.encoder = ..

        elif (encoder == "transformer"):
            self.encoder = TransformerInterEncoder(self.hidden_size, self.ff_size,
                                                   self.heads,
                                                   self.dropout, self.inter_layers)
        elif (encoder == "rnn"):
            pass

        elif (encoder == "baseline"):
            pass

        else:
            raise ValueError("encoder must be classifier, transformer, rnn or baseline")

        # Start of Keras Functional API #

        input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
        segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
        mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
        cls_ids = tf.keras.layers.Input(shape=(100,), dtype=tf.int32)
        mask_cls = tf.keras.layers.Input(shape=(100,), dtype=tf.int32)

        helper = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

        inputs = [input_ids, segment_ids, cls_ids, mask, mask_cls, helper]

        res = self.bert(inputs, training=True)

        sent_scores = self.encoder(res, mask_cls)

        outputs = sent_scores

        opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, lr=5e-5)

        # define the loss function and optimizer #
        def custom_loss_wrapper(inputs):

            def custom_loss(y_true, y_pred):
                mask_cls = inputs[-1]
                bce = tf.keras.losses.BinaryCrossentropy()
                tf.dtypes.cast(y_true, dtype=float)
                loss = bce(y_true, y_pred)
                tf.dtypes.cast(mask_cls, dtype=float)
                loss = (loss * mask_cls).sum()

                return loss

        # finish define custom loss function #
        bce1 = tf.keras.losses.BinaryCrossentropy()
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BertSum")
        # model.compile(optimizer=opt, loss=custom_loss_wrapper(inputs=inputs), run_eagerly=True)
        model.compile(optimizer=opt, loss=bce1, run_eagerly=False)

        # return #
        self.model = model

        # self.model.summary()

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    # Get the batch data
    def generate_train_batch(self, x_samples, batch_size):

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
                labels_batch = x_samples[1][start:end]
                segment_ids_batch = x_samples[2][start:end]
                cls_ids_batch = x_samples[3][start:end]
                mask_cls_batch = x_samples[4][start:end]
                mask_batch = x_samples[5][start:end]
                helper_batch = np.arange(0, batch_size, dtype=int)[:, None]

                x = [input_ids_batch, segment_ids_batch, cls_ids_batch, mask_batch, mask_cls_batch, helper_batch]

                yield x, labels_batch

    def generate_val_batch(self, x_samples, batch_size):

        num_batches = len(x_samples) // batch_size
        if num_batches == 0:
            raise ValueError("ddd")
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                input_ids_batch = x_samples[0][start:end]
                labels_batch = x_samples[1][start:end]
                segment_ids_batch = x_samples[2][start:end]
                cls_ids_batch = x_samples[3][start:end]
                mask_cls_batch = x_samples[4][start:end]
                mask_batch = x_samples[5][start:end]

                yield [input_ids_batch, segment_ids_batch, cls_ids_batch, mask_batch, mask_cls_batch], labels_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, )

    # Use fit_generator to train the model
    def fit(self, Xtrain, Xval, epochs=None, batch_size=None, model_dir_path=None):

        if epochs is None:
            epochs = NUM_TRAIN_EPOCHS

        if batch_size is None:
            batch_size = BATCH_SIZE

        if model_dir_path is None:
            model_dir_path = "./models"

        checkpoint_filepath = '/tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        train_generator = self.generate_train_batch(Xtrain, batch_size)
        validation_generator = self.generate_val_batch(Xval, batch_size)

        num_train_steps = len(Xtrain[0]) // batch_size

        if Xval is not None:
            num_validation_steps = len(Xval) // batch_size

        history = self.model.fit(x=train_generator, epochs=epochs,
                                           verbose=1, validation_data=None,
                                           validation_steps=None,
                                           steps_per_epoch = num_train_steps,
                                           callbacks=None)

        # self.model.save("BertSum")
        # self.model.load_weights(checkpoint_filepath)
        return history

    def test(self):
        pass

    def summarize(self, input):
        # preprocess the input text into [input_ids, segment_ids, cls_ids, mask, mask_cls]
        if len(input) == 0:
            return None
        else:
            config = BertConfig.from_pretrained(self.bert_name)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=self.bert_name, config=config)
            self.point_id = self.tokenizer.vocab(".")
            self.sep_vid = self.tokenizer.vocab("[SEP]")
            self.cls_vid = self.tokenizer.vocab("[CLS]")
            self.pad_vid = self.tokenizer.vocab("[PAD]")
            # ========add CLS and SEP between sent============#
            src = []
            src_str = []
            sent = []
            for char in input:
                if char == " ":
                    sent.append(char)
                elif char == ".":
                    src.append(char + "[SEP][CLS]")
                    sent.append(char)
                    src_str.append(sent)
                    sent = []
                else:
                    src.append(char)
                    sent.append(char)

            src_text = " ".join(src)
            src_tokens = self.tokenizer.tokenize(src_text)
            src_tokens = src_tokens[:510]
            src_tokens = ["[CLS]"] + src_tokens + "[SEP]"
            src_tokens_idxs = self.tokenizer.convert_tokens_to_ids(src_tokens)

            _segs = [-1] + [i for i, t in enumerate(src_tokens_idxs) if t == self.sep_vid]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

            segments_ids = []

            for i, s in enumerate(segs):
                if (i % 2 == 0):
                    segments_ids += s * [0]
                else:
                    segments_ids += s * [1]

            cls_ids = [i for i, t in enumerate(src_tokens_idxs) if t == self.cls_vid]
            mask_cls = []
            mask = []

            padding_length_subtoken = 512 - len(src_tokens_idxs)
            if padding_length_subtoken > 0:
                src_tokens_idxs = src_tokens_idxs + ([self.pad_vid] * padding_length_subtoken)
                segments_ids = segments_ids + ([0] * padding_length_subtoken)
                mask = (src_tokens_idxs != self.pad_vid)

            padding_length_sent = 100 - len(cls_ids)
            if padding_length_sent > 0:
                cls_ids = cls_ids + ([-1] * padding_length_sent)
                mask_cls = (cls_ids != -1)
                cls_ids[cls_ids == -1] = 0

            # =======create input====== #
            inputs = [src_tokens_idxs, segments_ids, cls_ids, mask, mask_cls]

            for x in inputs:
                x = np.array(x)

            # =========pred========== #

            sent_scores = self.model.predict(inputs)
            tf.dtypes.cast(mask, dtype=float)
            sent_scores = sent_scores + mask
            # print(sent_scores)
            selected_ids = np.argsort(-sent_scores, 1)
            # selected_ids 是每个cls的位置

            pred = []

            for i, idx in enumerate(selected_ids):
                _pred = []
                cls_idx = idx
                for i in range(cls_idx, 512):
                    if src_tokens[i] == "[SEP]":
                        _pred.append(src_tokens[i])
                        pred.append(_pred)
                        break
                    else:
                        _pred.append(src_tokens[i])

            ans = [" ".join(_pred) for _pred in pred]

            return ans[0]
