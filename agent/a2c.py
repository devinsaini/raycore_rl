import tensorflow as tf
from agent.model import Actor, Critic
from agent.actorcritic import ActorCritic

class A2CAgent(ActorCritic):
    def __init__(self, config) -> None:
       super().__init__(config)

