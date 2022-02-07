import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

rewards = []
with open("../train.log","r") as f:
    data = f.readlines()
    
for line in data:
    if line[0] == "e":
        reward = eval(line.split("=")[-1])
        rewards.append(reward)


x = np.arange(len(rewards))
rewards = np.array(rewards)
plt.title("train and eval")
plt.plot(x, rewards, label="train")

test_rewards_x = []
test_rewards_y = []
dir_ = os.listdir("../save/")
dir_ = sorted(dir_, key=lambda x : int(x.split("_")[0].split("-")[1]))
for file in dir_:
    x = int(file.split("_")[0].split("-")[1])
    y = float(file.split("_")[-1][:-5])
    test_rewards_x.append(x)
    test_rewards_y.append(y)

print(sum(rewards > 50))

plt.plot(test_rewards_x, test_rewards_y, label="test")
plt.legend()
plt.tight_layout()
plt.savefig("train_log.png")
# plt.show()
