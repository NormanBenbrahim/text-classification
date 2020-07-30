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
    def __init__(self, pre_trained_model, epochs, batch_size):
        self.pre_trained_model = pre_trained_model
        self.epochs = epochs
        self.batch_size = batch_size

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

        # create validation set by splitting 10k examples
        # from the original data
        x_val = train_examples[:10000]
        partial_x_train = train_examples[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]

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

            # build the keras layer with pretrained vectors
            # to map sentences into embedding vectorss
            hub_layer = hub.KerasLayer(model_url, output_shape=[output_shape],
                                       dtype=tf.string, trainable=True)

            print("")
            print("")
            print(hub_layer(train_examples[:3]))

            # initiate the model
            model = tf.keras.Sequential()
            # hub layer for embeddings
            model.add(hub_layer)
            # add a dense layer with 16 units
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            # now a single layer for output, log-likelihood default
            model.add(tf.keras.layers.Dense(1))

            # print a summary of the model
            print("")
            model.summary()

            print("")
            print("Compiling model")
            # now compile the model with binary crossentropy
            # as a loss function
            model.compile(optimizer='adam',
                          loss=tf.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            
            print("")
            print("Training the model")
            # train the model for user specified number of 
            # batches and samples
            history = model.fit(partial_x_train,
                                partial_y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(x_val, y_val),
                                verbose=1)
            
            print("")
            print("Evaluating the model")
            return model 

        except Exception as e:
            print(e)


if __name__ == '__main__':
    model = Model(pre_trained_model='nnlm128',
                  batch_size=512,
                  epochs=40)
    model.build_model()