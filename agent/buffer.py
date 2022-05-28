class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.returns = None
        self.advantages = None
    
    def add(self, state, action, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.returns = []
        self.advantages = []

def merge_buffers(buffers):
    states = []
    actions = []
    returns = []
    advantages = []
    for buffer in buffers:
        states.extend(buffer.states)
        actions.extend(buffer.actions)
        returns.extend(buffer.returns)
        advantages.extend(buffer.advantages)
    return states, actions, returns, advantages