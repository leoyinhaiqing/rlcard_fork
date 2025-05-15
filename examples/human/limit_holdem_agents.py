''' A toy example of playing against a random agent on Limit Hold'em
'''
import time

import rlcard
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.agents import RandomAgent
from rlcard.utils.utils import print_card

import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Make environment
env = rlcard.make(env_id='limit-holdem', config={"num_players": 6})
human_agent = HumanAgent(env.num_actions)
agent_0 = RandomAgent(num_actions=env.num_actions)
agent_1 = RandomAgent(num_actions=env.num_actions)
agent_2 = RandomAgent(num_actions=env.num_actions)
agent_3 = RandomAgent(num_actions=env.num_actions)
agent_4 = RandomAgent(num_actions=env.num_actions)

model_file_temp = "../experiments/limit-holdem_dqn_500_result/model.pth"
agent_dqn_500 = torch.load(model_file_temp)
model_file = "../experiments/limit-holdem_dqn_1000_result/model.pth"
agent_dqn_1000 = torch.load(model_file)
model_file2 = "../experiments/limit-holdem_dqn_5000_result/model.pth"
agent_dqn_5000 = torch.load(model_file2)
model_file3 = "../experiments/limit-holdem_dqn_10000_result/model.pth"
agent_dqn_10000 = torch.load(model_file3)
agent_dqn4 = torch.load(model_file)
agent_dqn5 = torch.load(model_file)
agent_dqn6 = torch.load(model_file)

model_file_nfsp = "../experiments/limit-holdem_nfsp_5000_result/model.pth"
agent_nfsp = torch.load(model_file_nfsp)


env.set_agents([
    # human_agent,
    agent_dqn_500,
    agent_dqn_1000,
    agent_dqn_5000,
    agent_dqn_10000,
    agent_2,
    agent_3,
])

print(">> Limit Hold'em random agent")

df_reward = pd.DataFrame(columns=list(range(env.num_players)))
df_reward.loc[0] = np.zeros(env.num_players)
round = 0
while (round < 300):
    print(">> Start a new game")
    round = round + 1
    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    if len(trajectories[0]) != 0:
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            """
            if action_record[-i][0] == state['current_player']:
                break
            """
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    # print('=============     Random Agent    ============')
    # print_card(env.get_perfect_information()['hand_cards'][1])
    print('=============     DQN             =============')
    print_card(env.get_perfect_information()['hand_cards'][0])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    # df_reward.loc[len(df_reward), "reward"] = payoffs[0]

    for i in range(0, env.num_players):
        df_reward.loc[round, i] = payoffs[i]


    if round % 10 == 0:
        plt.figure(figsize=(12, 8))
        cum_reward = df_reward.values.cumsum(axis=0)
        df_cum_reward = pd.DataFrame(cum_reward, columns=list(range(env.num_players)))
        # plt.plot(cum_reward, color="darkred", label="DQN Agent")
        df_cum_reward.plot(figsize=(12, 8))
        plt.title("DQN vs Random Agents")
        plt.legend()
        plt.tight_layout()
        plt.savefig('dqn_reward.png')
        df_reward.to_csv('reward.csv', index=False)

    time.sleep(0.2)
    # input("Press any key to continue...")

# import matplotlib.pyplot as plt
# import pandas as pd
#
# df_reward = pd.read_csv('reward.csv')
# plt.figure(figsize=(12, 8))
# cum_reward = df_reward['reward'].cumsum()
# plt.plot(cum_reward, color="darkred", label="DQN Agent")
# plt.title("DQN vs Random Agents")
# plt.legend()
# plt.tight_layout()
# plt.savefig('dqn_reward.png')
# df_reward.to_csv('reward.csv', index=False)
