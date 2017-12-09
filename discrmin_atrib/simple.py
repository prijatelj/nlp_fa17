"""
NN test bed for capturing discriminative attributes between words.

Example on how to run this with flags in cli:

python simple.py --train_embed=True --epochs=100 --batch_size=100 `
--hidden_layers=25 --hidden_nodes=200 --model="dense"

Extra or specific flags:
--dropout_rate=0.0 --learning_rate=1e-3

@author: Derek S. Prijatelj
"""

from datetime import datetime
from neural_nets import sts_data_handler, embed_models
from neural_nets import main as nn_main
from neural_nets.simple_nn_models import lstm, dense, conv

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold

if "timestamp" not in tf.flags.FLAGS.__dict__['__flags']:
    tf.flags.DEFINE_boolean("timestamp", True,
                            "appends date and time to model name")

tf.flags.FLAGS._parse_flags()
FLAGS = tf.flags.FLAGS

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
# python simple.py --flag_name=VALUE ...
print("Train, Evalutate, Test: ", MODEL_NAME)

def init(kfolds=FLAGS.kfolds, final_train_test=True):
    # get data
    train_w1, train_w2, train_feature, train_labels = sts_data_handler. \
        read_data(
            "data/DiscriminAtt/training/train.txt"
        )
    test_w1, test_w2, test_feature, test_labels = sts_data_handler.read_data(
        "data/DiscriminAtt/training/validation.txt"
        )

    embed_index = sts_data_handler.embed_index("data/numberbatch-en-17.06.txt")

    # Make Model
    # embed
    w1 = np.append(train_w1, test_w1)
    w2 = np.append(train_w2, test_w2)
    feats = np.append(train_feature, test_feature)
    #embed_model, emb_w1, emb_w2, emb_feat = embed_models.word_embed_tokenizer(
    #    w1, w2, feats, embed_index)
    embed_model, emb_data = embed_models.word_embed_tokenizer(
        w1, w2, feats, embed_index)

    # split apart the embedding to obtain data split padded sequences
    train_data = emb_data[:len(train_feature)]
    test_data = emb_data[len(train_feature):]

    print("after emb_data train_data = ", len(train_data))
    print("after emb_data test_data = ", len(test_data))

    train_labels = to_categorical(train_labels, 2)
    test_labels = to_categorical(test_labels, 2)

    if kfolds >= 2:
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True)
        for fold_num, train_index, test_index in enumerate(
                skf.split(train_data, train_labels)):
            x_data = train_data[train_index]
            x_labels = train_labels[train_index]
            y_data = train_data[test_index]
            y_labels = train_labels[test_index]

            print("\nk-fold # ", fold_num, "\n")
            train_test(x_data, x_labels,
                       y_data, y_labels,
                       embed_model,
                       save_model=False,
                       final_test=False)

    if final_train_test:
        print("\nTEST:\n")
        train_test(
            train_data, train_labels,
            test_data, test_labels,
            embed_model
        )

def train_test(train_data, train_labels,
               test_data, test_labels,
               embed_model,
               save_model=True,
               load_train=False,
               final_test=True
              ):
    # Classification Model
    if FLAGS.model == "lstm":
        classification_model = lstm(
            train_data.shape, embed_model, 1)
    elif FLAGS.model == "dense":
        classification_model = dense(
            train_data.shape, embed_model, 1)
    elif FLAGS.model == "conv":
        classification_model = conv(
            train_data.shape, embed_model, 1)

    print("\n train_data length ", len(train_data))
    print("\n train_labels length ", len(train_labels))
    print("\n test_data length ", len(test_data))
    print("\n test_labels length ", len(test_labels))

    # Train
    if final_test:
        results_dict, trained_model = nn_main.train_and_eval(
            classification_model,
            train_data,
            train_labels,
            test_data,
            test_labels)
    else:
        results_dict, trained_model = nn_main.train_and_eval(
            classification_model,
            train_data,
            train_labels)

    # Save model
    if save_model:
        trained_model_json = trained_model.to_json()
        with open("trained_model_json/" + MODEL_NAME + ".json", "w") as \
                json_file:
            json_file.write(trained_model_json)
        trained_model.save_weights("trained_model_h5/" + MODEL_NAME +".h5")

    if final_test:
        test(trained_model, test_data, test_labels)

def load_test():
    # Load Existing Trained Model
    with open("trained_modeltrained_via_dev_1epoch_longrun.json", 'r') \
            as json_file:
        trained_model_json = json_file.read()
    trained_model = model_from_json(trained_model_json)
    trained_model.load_weights(
        "trained_model_trained_via_dev_1epoch_longrun.h5")
    trained_model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"])

    test(trained_model, test_data, test_labels)

def test(trained_model, test_data, test_labels):
    # Test
    def test_eval(model, test_data, test_labels):
        eval_results = model.evaluate(test_data,
                                      test_labels)
        pred = np.squeeze(model.predict(test_data,
                                        batch_size=FLAGS.batch_size))

        eval_metrics = model.metrics_names
        results_dict = dict(zip(eval_metrics, eval_results))
        return results_dict, pred

    test_results_dict, pred = test_eval(trained_model, test_data, test_labels)
    print("Results:")
    with open("results/" + MODEL_NAME + "_TestResults.txt", "w") as res_file:
        for k, v in test_results_dict.items():
            res_file.write(str(k) + " = " + str(v) + "\n")
            print(k, " = ", v)

    # save output file
    with open("results/predictions/" + MODEL_NAME + "_TestPredictions.txt",
            "w") as pred_file:
        for p in pred:
            #print(np.array_str(p))
            p = p.tolist()
            pred_file.write((str)(p.index(max(p))) + "\n")

def main(argv):
    init()

if __name__ == "__main__":
    tf.app.run()
