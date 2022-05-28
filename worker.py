import ray
import gym
from agent.buffer import RolloutBuffer


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, agent_cls, configs, env=None) -> None:
        self.env    = gym.make(env)
        self.agent  = agent_cls(configs)
        self.buffer = RolloutBuffer()

    def play_episode(self, max_t, weights=None):
        self.agent.set_weights(weights[0], weights[1])
        state = self.env.reset()
        self.buffer.reset()
        done  = False
        
        for t in range(max_t):
            action, _, value = self.agent.act(state, training=True)
            action = action.numpy()[0]
            value = value.numpy()[0]
            next_state, reward, done, _ = self.env.step(action)

            self.buffer.add(state, action, reward, value, done)
            state = next_state

            if done:
                _, _, final_val = self.agent.act(state, training=True)
                self.buffer.values.append(final_val.numpy()[0])
                break

        self.buffer.returns    = self.agent.calc_gae_returns(self.buffer.rewards, self.buffer.dones, self.buffer.values)
        self.buffer.values     = self.buffer.values[:-1]
        self.buffer.advantages = self.buffer.returns - self.buffer.values

        return self.buffer


@ray.remote(num_cpus=1, num_gpus=1)
class Trainer:
    def __init__(self, agent_cls, configs) -> None:
        self.agent = agent_cls(configs)

    def train(self, states, actions, returns, advantages):
        self.agent.train(states, actions, returns, advantages)

    def get_weights(self):
        return self.agent.get_weights()