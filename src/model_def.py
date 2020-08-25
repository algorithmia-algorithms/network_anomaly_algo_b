import shutil
import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = layers.Dense(4, name="layer1", activation="relu")
        self.layer2 = layers.Dense(4, name="layer2", activation="relu")
        self.out = layers.Dense(1, name="output")

    def call(self, inputs, training=None, mask=None):
        o1 = self.layer1(inputs)
        o2 = self.layer2(o1)
        final = self.out(o2)
        return final


def main():
    model = Model()
    model.build((1, 5))
    test_input = tf.constant([[4.0, 3.0, 2.2, -1., 0.25]], shape=(1, 5))
    test_output = model.predict(test_input)
    print(test_output)
    model.save("/tmp/tf_model")
    shutil.make_archive("/tmp/tf_model_example", 'zip', "/tmp/tf_model")


main()
