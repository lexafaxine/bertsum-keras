from __future__ import print_function
import tensorflow as tf
import numpy as np
from model import create_model
from preprocess import BertData, Processor
from data_loader import transfer_examples_to_inputs


def get_predict_input(input_data):

    input_ids = input_data[0]
    labels = input_data[-1]
    segment_ids = input_data[1]
    cls_ids = input_data[3]
    mask_cls = input_data[4]
    mask = input_data[2]
    helper = np.arange(0, 1, dtype=int)[:, None]

    x = [input_ids, segment_ids, cls_ids, mask, mask_cls, helper, labels]

    return x

if __name__ == "__main__":
    ckpt_path="./training_1/cp.ckpt"
    # load model
    bert_sum = create_model("bert-base-uncased", 512, "transformer", 768, 512, 8, 0.1, 2)
    bert_sum.load_weights(ckpt_path)

    # # show the model
    bert_sum.summary()

    # # check the accuracy and loss
    # loss, acc = bert_sum.evaluate()
    text = "Two New York doormen who closed the doors to their building's apartment lobby while a 65-year-old Asian woman was punched and kicked outside have been fired, according the building's owner.The Brodsky Organization, which owns the building at 360 W. 43rd St. in Manhattan, said in a statement Tuesday that it had completed an inquiry into the doormen's response to the March 29 attack."

    # input_text, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, max_length
    Processor = Processor("bert-base-uncased", 3, 200, 100, 3, 512, "predict")
    example, src = Processor.preprocess(text, "", oracle_ids=[])
    input = transfer_examples_to_inputs([example], is_test=False)
    #
    x = get_predict_input(input)
    # print(x)
    result = bert_sum.predict(x)
    print("Result:", result)
    rank = 3
    ind = np.argsort(-result).astype(np.int32)
    print("ind", ind)

    for i in range(rank):
        print("Summ",i,": ", src[ind[0][i]])