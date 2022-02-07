import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class GameModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(GameModel, self).__init__()
        hid_size = act_dim * 20
        self.fc1 = nn.Linear(obs_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, hid_size)
        self.fc4 = nn.Linear(hid_size, hid_size)
        self.fc5 = nn.Linear(hid_size, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        Q = self.fc5(x)
        return Q
