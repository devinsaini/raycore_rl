from worker import Worker, Trainer
import ray
import numpy as np
from agent.buffer import merge_buffers
from agent.a2c import A2CAgent
import gym

ENV_NAME    = "CartPole-v0"
NUM_WORKERS = 8
N_EPOCHS    = 50
MAX_T       = 200
AGENT_CONF  = {
    'num_actions'  : 2,
    'state_dim'    : 4,
    'value_coeff'  : 0.5,
    'entropy_coeff': 0.01,
    'gamma'        : 0.99,
    'tau'          : 0.95,
    'lr'           : 0.005
}

def main():
    ray.init(num_cpus=NUM_WORKERS+1, num_gpus=1)

    # create remote workers
    trainer = Trainer.remote(A2CAgent, AGENT_CONF)
    workers = [Worker.remote(A2CAgent, AGENT_CONF, ENV_NAME) for _ in range(NUM_WORKERS)]
    
    weights = trainer.get_weights.remote()

    # play episodes
    for i in range(N_EPOCHS):
        buffer_refs = [worker.play_episode.remote(MAX_T, weights) for worker in workers]

        # merge buffers
        buffers = ray.get(buffer_refs)
        states, actions, returns, advantages = merge_buffers(buffers)

        # train
        trainer.train.remote(states, actions, returns, advantages)
        weights = trainer.get_weights.remote()

        # show training progress
        score     = np.mean([len(buffer.rewards) for buffer in buffers])
        min_score = np.min([len(buffer.rewards) for buffer in buffers])
        max_score = np.max([len(buffer.rewards) for buffer in buffers])
        print("Episode:{:4d},\tscore - min:{:3.2f},\tavg:{:3.2f},\tmax:{:3.2f},\tsamples in batch:{:4d}".format(i, min_score, score, max_score, len(states)))

    # evaluation run
    env   = gym.make(ENV_NAME)
    agent = A2CAgent(AGENT_CONF)
    agent.set_weights(*ray.get(weights))
    for i in range(5):
        evaluate(env, agent)

    ray.shutdown()
    

def evaluate(env, agent):
    done  = False
    state = env.reset()
    score = 0
    while not done:
        action, _, _ = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action.numpy()[0])
        score += reward

    print("Evaluation score: {:3.2f}".format(score))
    

if __name__ == "__main__":
    main()