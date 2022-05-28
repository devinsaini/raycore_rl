import tensorflow as tf
import tensorflow_probability as tfp
from agent.model import Actor, Critic

class PPOAgent:
    def __init__(self, config) -> None:
       self.num_actions = config['num_actions']
       self.state_dim   = config['state_dim']

       self.actor       = Actor(self.num_actions)
       self.critic      = Critic()

    
    def act(self, state):
        state         = tf.expand_dims(state, axis=0)
        action_logits = self.actor(state)
        action_dist   = tfp.distributions.Categorical(logits=action_logits)
        action        = action_dist.sample()

        value         = self.critic(state)
        return action, value