import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal

def build_dqn(lr, n_actions, input_dims, fcl_dims, fc2_dims):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(fcl_dims, input_shape=(input_dims, ), activation='relu'),
        tf.keras.layers.Dense(fc2_dims, activation='relu'),
        tf.keras.layers.Dense(n_actions)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='mse')

    return model

class Agent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_min=0.01, mem_size=1000000, model_file='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = model_file

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = np.random.choice(self.action_space) if np.random.random() < self.epsilon else np.argmax(self.q_eval.predict(state))
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indicies = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indicies] = reward + self.gamma * np.max(q_next, axis=1) * done
        
        self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = tf.keras.models.load_model(self.model_file)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=0, alpha=0.0005, input_dims=8, n_actions=4, mem_size=1000000, batch_size=64, epsilon_dec=1, epsilon_min=0)

    agent.load_model()

    while input('Would you like to watch the AI play another game (Y/N)').lower() == 'y':
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            score += reward
            env.render()

        print("Score was", score)

# def train():
#     env = gym.make('LunarLander-v2')
#     n_games = 500
#     agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=8, n_actions=4, mem_size=1000000, batch_size=64, epsilon_dec=0.999, epsilon_min=0.01)

#     scores = []
#     eps_history = []

#     for i in range(n_games):
#         done = False
#         score = 0
#         observation = env.reset()

#         while not done:
#             action = agent.choose_action(observation)
#             next_observation, reward, done, info = env.step(action)
#             score += reward
#             agent.remember(observation, action, reward, next_observation, done)
#             observation = next_observation
#             agent.learn()

#         eps_history.append(agent.epsilon)
#         scores.append(score)

#         avg_score = np.mean(scores[max(0, i - 100): i + 1])
#         print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

#         if i % 10 == 0 and i > 0:
#             agent.save_model()

#         plt.plot(scores)
#         plt.savefig('Scores.png')
#         plt.plot(eps_history)
#         plt.savefig('Epsilon History.png')