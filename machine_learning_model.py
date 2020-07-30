import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

# build the model
# based on https://www.tensorflow.org/hub/tutorials/tf2_text_classification
class Model():
    """

    """
    def __init__(self, pre_trained_model):
        self.pre_trained_model = pre_trained_model

    def build_model(self):
        """
        load pre-trained text embeddings as 
        first layer using tf hub
        see 
        """

        ## use imdb data from tf datasets for now
        train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

        train_examples, train_labels = tfds.as_numpy(train_data)
        test_examples, test_labels = tfds.as_numpy(test_data)

        # assert model type 
        try:
            if self.pre_trained_model == 'swivel20_oov':
                model_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
                output_shape = 20

            elif self.pre_trained_model == 'swivel20' :
                model_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
                output_shape = 20

            elif self.pre_trained_model == 'nnlm50':
                model_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
                output_shape = 50
            elif self.pre_trained_model == 'nnlm128':
                model_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
                output_shape = 128
            else:
                print("You did not select a valid model")
                print("Valid options: swivel20_oov, swivel20, nnlm50, nn;m128")
                raise Exception

            hub_layer = hub.KerasLayer(model_url, output_shape=[output_shape],
                                       dtype=tf.string, trainable=True)

            hub_layer(train_examples[:3])
            

        except Exception as e:
            print(e)


if __name__ == '__main__':
    model = Model(pre_trained_model='nnlm128')
    model.build_model()