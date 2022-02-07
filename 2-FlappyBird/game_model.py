import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class GameModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(GameModel, self).__init__()

        self.conv1 = nn.Conv2D(4, 32, (3, 3),padding="SAME")
        self.conv2 = nn.Conv2D(32, 32, (3, 3),padding="SAME")
        self.conv3 = nn.Conv2D(32, 64, (3, 3),padding="SAME")
        self.conv4 = nn.Conv2D(64, 64, (3, 3),padding="SAME")
        self.fc1 = nn.Linear(64 * 25, 256)
        self.fc2 = nn.Linear(256, act_dim)
        self.flatten = nn.Flatten()

    def forward(self, obs):
        x = obs
        # 40, 20, 10, 5
        for conv in [self.conv1, self.conv2,self.conv3, self.conv4]:
            x = conv(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
            x = F.relu(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import numpy as np
    model = GameModel(None,2)
    input_data = np.random.rand(1, 4, 80, 80).astype('float32')
    x = paddle.to_tensor(input_data)
    out = model(x)
    print(out)
