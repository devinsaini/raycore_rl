import tensorflow as tf
from tensorflow.keras import layers

class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.fc1 = layers.Dense(units=64, activation='tanh')
        self.fc2 = layers.Dense(units=64, activation='tanh')
        self.fc3 = layers.Dense(units=action_dim)

    def call(self, inputs, training=False):
        y             = self.fc1(inputs, training=training)
        y             = self.fc2(y, training=training)
        action_logits = self.fc3(y, training=training)

        return action_logits


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(units=64, activation='tanh')
        self.fc2 = layers.Dense(units=64, activation='tanh')
        self.fc3 = layers.Dense(units=1)

    def call(self, inputs, training=False):
        y            = self.fc1(inputs, training=training)
        y            = self.fc2(y, training=training)
        critic_value = self.fc3(y, training=training)
        critic_value = tf.squeeze(critic_value, axis=-1)
        
        return critic_value
