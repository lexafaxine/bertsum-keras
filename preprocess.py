from __future__ import print_function
import re
from transformers import BertTokenizer, TFBertModel, BertConfig



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


class Processor(object):
    def __init__(self, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, max_length, mode):

        self.min_src_ntokens = min_src_ntokens
        self.max_src_ntokens = max_src_ntokens
        self.max_nsents = max_nsents
        self.min_nsents = min_nsents
        self.max_length = max_length
        self.mode = mode

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
        # oracle_ids is the extractive summarization of the text

        mode = self.mode

        if (len(src) == 0):
            return None

        if mode == "predict":
            input_text = src
            src = []
            n = len(input_text)
            # find the sentence
            start = 0
            end = 0
            for i in range(n):
                if input_text[i] == ".":
                    end = i
                    # print("end=", end)
                    sent_text = input_text[start:end]
                    # print("sent_text:", sent_text)
                    sent = sent_text.split()
                    sent.append(".")
                    src.append(sent)
                    # print(sent)
                    start = i + 1
            print(src)

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
        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP][CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:self.max_length - 2]

        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        # get the length of each sentence
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []

        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]
        assert len(cls_ids) == len(labels)

        if sum(labels) == 0 and mode == "train":
            # all elements in labels is 0: dont make sense
            return None


        if mode == "train":
            tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])

        if mode == "predict":
            tgt_txt = ""

        src_txt = [original_src_txt[i] for i in idxs]

        # ======================PAD=========================== #

        mask_cls = [0 for _ in range(self.max_nsents)]
        mask = [0 for _ in range(self.max_length)]

        padding_length_subtoken = self.max_length - len(src_subtoken_idxs)
        if padding_length_subtoken > 0:
            src_subtoken_idxs = src_subtoken_idxs + ([self.pad_vid] * padding_length_subtoken)
            segments_ids = segments_ids + ([0] * padding_length_subtoken)

        for i in range(0, self.max_length):
            if src_subtoken_idxs[i] != self.pad_vid:
                mask[i] = 1
            else:
                mask[i] = 0

        padding_length_sent = self.max_nsents - len(labels)
        if padding_length_sent > 0:
            labels = labels + ([0] * padding_length_sent)
            cls_ids = cls_ids + ([-1] * padding_length_sent)

        assert len(cls_ids) == len(labels) == len(mask_cls)

        for i in range(0, 100):
            if cls_ids[i] == -1:
                mask_cls[i] = 0
            else:
                mask_cls[i] = 1

        for i in range(0, 100):
            if cls_ids[i] == -1:
                cls_ids[i] = 0

        cnn_data = BertData(input_ids=src_subtoken_idxs, labels=labels, segments_ids=segments_ids, cls_ids=cls_ids,
                            mask_cls=mask_cls, mask=mask, src_txt=src_txt,
                            tgt_txt=tgt_txt)
        if mode == "predict":
            return cnn_data, src


        return cnn_data