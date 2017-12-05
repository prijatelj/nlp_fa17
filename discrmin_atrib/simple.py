"""
Hopefully simple implementation of an LSTM for multi classification of Emoji
prediciton given tweet text.

Example on how to run this with flags in cli:

python simple.py --train_embed=True --epochs=100 --batch_size=100 `
--hidden_layers=25 --hidden_nodes=200 --model="dense"

Extra or specific flags:
--dropout_rate=0.0 --learning_rate=1e-3

@author: Derek S. Prijatelj
"""

from datetime import datetime
from neural_nets import sts_data_handler, embed_models, main
from neural_nets.simple_nn_models import lstm, dense, conv
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from keras.models import model_from_json
from scipy.stats import pearsonr

if "timestamp" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("timestamp", True,
                            "appends date and time to model name")

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

# python simple.py --flag_name=VALUE ...
MODEL_NAME = "" if FLAGS.model_id == "" else FLAGS.model_id + "_"
MODEL_NAME +=(
              "Model-" + str(FLAGS.model)
              +"_HiddenLayers-" + str(FLAGS.hidden_layers)
              +"_HiddenNodes-" + str(FLAGS.hidden_nodes)
              +"_TrainEmbed-" + str(FLAGS.train_embed)
              +"_LearnRate-" + str(FLAGS.learning_rate)
              +"_Epochs-" + str(FLAGS.epochs)
              +"_BatchSize-" + str(FLAGS.batch_size)
             )

if FLAGS.time_steps != 1:
    MODEL_NAME += "_TimeStep-" + str(FLAGS.time_steps)

if FLAGS.dropout_rate != 0:
    MODEL_NAME += "_DropoutRate-" + str(FLAGS.dropout_rate)

if FLAGS.timestamp:
    MODEL_NAME += "__" + \
        str(datetime.now()).replace(' ', "-at-").replace(":", "-")

print("Train, Evalutate, Test: ", MODEL_NAME)

# get data
train_texts, train_labels = sts_data_handler.read_data(
    #"data/sml_train_text.txt",
    #"data/sml_train_labels.txt"
    "data/ftrain_text.txt",
    "data/ftrain_labels.txt"
    #"data/fdev_text.txt",
    #"data/fdev_labels.txt"
    )

dev_texts, dev_labels = sts_data_handler.read_data(
    #"data/sml_dev_text.txt",
    #"data/sml_dev_labels.txt"#
    "data/fdev_text.txt",
    "data/fdev_labels.txt"
    )

test_texts, test_labels = sts_data_handler.read_data(
    #"data/sml_test_text.txt",
    #"data/sml_test_labels.txt"
    "data/ftest_text.txt",
    "data/ftest_labels.txt"
    )

embed_index = sts_data_handler.embed_index()

# Make Model
# embed
texts = np.append(train_texts, np.append(dev_texts, test_texts))
embed_model, pad_seq = embed_models.word_embed_tokenizer(texts,
                                                         embed_index)
embedded_sents = np.array(pad_seq)

# split apart the embedding to obtain data split padded sequences
dev_texts = embedded_sents[len(train_texts) : len(train_texts) + len(dev_texts)]
test_texts = embedded_sents[len(train_texts)+len(dev_texts):]
train_texts = embedded_sents[:len(train_texts)]

train_labels = to_categorical(train_labels, 20)
dev_labels = to_categorical(dev_labels, 20)
test_labels = to_categorical(test_labels, 20)
#"""
print("train_labels")
print(train_labels)

# Classification Model
if FLAGS.model == "lstm":
    classification_model = lstm(
        train_texts.shape, embed_model, dev_labels.shape[1])
elif FLAGS.model == "dense":
    classification_model = dense(
        train_texts.shape, embed_model, dev_labels.shape[1])
elif FLAGS.model == "conv":
    classification_model = conv(
        train_texts.shape, embed_model, dev_labels.shape[1])

# Train
results_dict, trained_model = main.train_and_eval(
    classification_model,
    train_texts,
    train_labels,
    dev_texts,
    dev_labels)

# Save model
trained_model_json = trained_model.to_json()
with open("trained_model_json/" + MODEL_NAME + ".json", "w") as json_file:
    json_file.write(trained_model_json)
trained_model.save_weights("trained_model_h5/" + MODEL_NAME +".h5")

"""
with open("trained_modeltrained_via_dev_1epoch_longrun.json", 'r') as json_file:
    trained_model_json = json_file.read()
trained_model = model_from_json(trained_model_json)
trained_model.load_weights("trained_model_trained_via_dev_1epoch_longrun.h5")
trained_model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
    metrics=["accuracy"])
#"""

# Test
def test_eval(model, test_data, test_labels):
    eval_results = model.evaluate(test_data,
                                  test_labels)
    pred = np.squeeze(model.predict(test_data,
                                    batch_size=FLAGS.batch_size))

    eval_metrics = model.metrics_names
    results_dict = dict(zip(eval_metrics, eval_results))
    return results_dict, pred

print("\nTEST:\n")
test_results_dict, pred = test_eval(trained_model, test_texts, test_labels)
#test_results_dict = test_eval(trained_model, test_texts, test_labels)
print("Test results:")
with open("results/" + MODEL_NAME + "_TestResults.txt", "w") as res_file:
    for k, v in test_results_dict.items():
        res_file.write(str(k) + " = " + str(v) + "\n")
        print(k, " = ", v)

# save output file
with open("results/predictions/" + MODEL_NAME + "_TestPredictions.txt", "w") as pred_file:
    for p in pred:
        #print(np.array_str(p))
        p = p.tolist()
        pred_file.write((str)(p.index(max(p))) + "\n")
