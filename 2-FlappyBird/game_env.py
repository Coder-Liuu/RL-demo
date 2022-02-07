from pygame.constants import K_w, K_s
from ple.games.flappybird import FlappyBird
from ple import PLE
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

class FlappyBird_Env():
    def __init__(self, IMAGE_SIZE):
        game = FlappyBird()
        self.env = PLE(game, fps=30, display_screen=True, force_fps=False)
        self.env.init()
        self.IMAGE_SIZE = IMAGE_SIZE

    def preprocess(self,image):
        image = cv2.resize(image, self.IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows = cols = self.IMAGE_SIZE[0]
        rotate = cv2.getRotationMatrix2D((rows*0.5, cols*0.5), 270, 1)
        image = cv2.warpAffine(image, rotate, (cols, rows))
        image = cv2.flip(image,1)  # 镜像


        image = np.expand_dims(image, axis=0)
        return image /255.0

    def get_process_image(self):
        screen = self.env.getScreenRGB()
        next_obs = self.preprocess(screen)
        self.count = 0
        return next_obs


    def step(self, action):
        reward = self.env.act(K_w) if action == 1 else self.env.act(action)
        done = True if self.env.game_over() else False

        # 重新计算reward
        if done:
            reward = -1
        elif reward == 1:
            reward = 1
            self.count += 1
        else:
            reward = 0.1

        next_obs = self.get_process_image()
        info = {}
        return next_obs, reward, done, info

    def reset(self):
        self.env.reset_game()
        obs = self.get_process_image()
        return obs

    def sample(self):
        action = random.choice([0,1])
        return action

if __name__ == "__main__":
    env = FlappyBird_Env(IMAGE_SIZE=(80,80))
    while True:
        env.reset()
        steps = 0
        total_reward = 0.0
        while True:
            action = env.sample()
            next_obs, reward, done, _ = env.step(action)
            next_obs = np.array(next_obs)[0]
            plt.figure()
            plt.imshow(next_obs)
            plt.savefig(f"image/game-{steps}.jpg")
            total_reward += reward
            steps += 1
            if done:
                exit(0)
                break
        print("total_reward = ", total_reward)
