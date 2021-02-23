from __future__ import print_function
import numpy as np
import os, logging
import json
from preprocess import Processor, greedy_selection


def create_examples(json_path, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, oracle_mode, mode,
                    max_length):

    if mode == "test":
        pass
    if mode == "val":
        pass

    if mode == "train":
        examples = []
        print("Preprocessing %s" % json_path)
        count = 0

        bert_data = Processor(bert_name=bert_name,
                              min_src_ntokens=min_src_ntokens,
                              max_src_ntokens=max_src_ntokens,
                              max_nsents=max_nsents,
                              min_nsents=min_nsents,
                              max_length=max_length)

        for i in range(0, 1):

            json_file = os.path.join(json_path, ".train." + str(i) + ".json")

            print("preprocessing " + json_file)

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

                    if count == 2000:
                        break

        print("Finish preprocessing")
        logging.info("Finish preprocessing")

        return examples

def transfer_examples_to_inputs(examples, is_test=False):
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
            dataset_dict["segments_ids"],
            dataset_dict["mask"],
            dataset_dict["cls_ids"],
            dataset_dict["mask_cls"],
            dataset_dict["src_txt"],
            dataset_dict["tgt_txt"],
            dataset_dict["labels"],
        ]
    else:
        inputs = [
            dataset_dict["input_ids"],
            dataset_dict["segments_ids"],
            dataset_dict["mask"],
            dataset_dict["cls_ids"],
            dataset_dict["mask_cls"],
            dataset_dict["labels"],
        ]

    return inputs

def create_inputs(json_path, bert_name, min_src_ntokens, max_src_ntokens, max_nsents, min_nsents, oracle_mode, mode,
                    max_length, is_test=False):
    examples = create_examples(json_path=json_path,
                               bert_name=bert_name,
                               min_src_ntokens=min_src_ntokens,
                               max_src_ntokens=max_src_ntokens,
                               max_nsents=max_nsents,
                               min_nsents=min_nsents,
                               oracle_mode=oracle_mode,
                               mode=mode,
                               max_length=max_length,)
    print("Totally ", len(examples), " examples.")
    inputs = transfer_examples_to_inputs(examples, is_test=is_test)

    return inputs
