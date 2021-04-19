In order to use the graph mode in Tensorflow2 to optimize the performance of original BertSum(https://github.com/nlpyang/BertSum), the original BertSum code is refactored in Tensorflow2 Keras API.

First of all, you need to preprocess the data and get the json file that is created in step3.(https://github.com/nlpyang/BertSum)

------

## Train the model

`python train_bertsum.py --json_path=YOUR_JSON_PATH --oracle_mode="greedy" --mode="train"`

and get the checkpoint files in "training_1/cp.ckpt"

------

## Predict

`python predict_bertsum.py`

------

To do:

Interface

Implement RNN and classifier and baseline