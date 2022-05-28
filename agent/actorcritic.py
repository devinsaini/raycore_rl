import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from agent.model import Actor, Critic


class ActorCritic:
    def __init__(self, config) -> None:
        self.num_actions   = config['num_actions']
        self.state_dim     = config['state_dim']
        self.entropy_coeff = config['entropy_coeff']
        self.value_coeff   = config['value_coeff']
        self.gamma         = config['gamma']
        self.tau           = config['tau']
        self.lr            = config['lr']

        self.actor     = Actor(self.num_actions)
        self.critic    = Critic()

        self.actor.build(input_shape=(None, self.state_dim))
        self.critic.build(input_shape=(None, self.state_dim))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)


    def act(self, state, training=False):
        state         = tf.expand_dims(state, axis=0)
        action_logits = self.actor(state)
        if training:
            action_dist   = tfp.distributions.Categorical(logits=action_logits)
            action        = action_dist.sample()
        else:
            action_dist   = None
            action        = tf.argmax(action_logits, axis=-1)

        value         = self.critic(state)
        return action, action_dist, value


    def calc_gae_returns(self, rewards, dones, values):
        trajectory_length = len(rewards)
        returns_arr       = tf.TensorArray(dtype=tf.float32, size=trajectory_length)

        gae = 0
        for i in tf.reverse(tf.range(trajectory_length), axis=[0]):
            terminal    = tf.cast(dones[i], tf.float32)
            delta       = rewards[i] + self.gamma * values[i+1] * (1.0 - terminal) - values[i]
            gae         = delta + self.gamma * self.tau * (1.0 - terminal) * gae
            returns_arr = returns_arr.write(i, gae + values[i])

        returns = returns_arr.stack()
        returns_arr.close()
        return returns


    def train(self, states, actions, returns, advantages):
        states = tf.convert_to_tensor(states)
        returns = tf.convert_to_tensor(returns)
        advantages = tf.convert_to_tensor(advantages)

        with tf.GradientTape() as tape:
            _, action_dist, value_p = self.act(states, training=True)

            policy_loss = tf.reduce_mean(-action_dist.log_prob(actions) * advantages)
            entropy = tf.reduce_mean(action_dist.entropy())
            value_loss = tf.reduce_mean(tf.square(returns - value_p))
            loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    
    def set_weights(self, actor_weights, critic_weights):
        self.actor.set_weights(actor_weights)
        self.critic.set_weights(critic_weights)


    def get_weights(self):
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        return actor_weights, critic_weights
