# Author: Shiva Verma
# 参考链接: https://github.com/shivaverma/Orbit
# 参考链接: https://github.com/cyfwry/Reinforcement-learning-algorithm-DQN-practice/blob/master/pad.py


import turtle as t
import numpy as np


class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # 设置背景

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # 飞浆

        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # 球

        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(np.random.randint(-290, 291), 100)
        self.ball.dx = np.random.randint(1, 6)*(2*np.random.randint(0, 2)-1)
        self.ball.dy = np.random.randint(-5, -2)

        # 分数

        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(
            self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # 飞浆移动

    def paddle_right(self):
        self.reward -= .1
        x = self.paddle.xcor()
        if x < 240:
            self.paddle.setx(x+20)

    def paddle_left(self):
        self.reward -= .1
        x = self.paddle.xcor()
        if x > -240:
            self.paddle.setx(x-20)

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):
        self.hit, self.miss = 0, 0
        self.paddle.goto(0, -275)
        self.ball.goto(np.random.randint(-290, 291), 100)
        self.pad = 0
        self.ball.dx = np.random.randint(1, 6)*(2*np.random.randint(0, 2)-1)
        self.ball.dy = np.random.randint(-5, -2)

        self.score.clear()
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        return [(self.paddle.xcor()+290)/580, (self.ball.xcor()+290)/580, (self.ball.ycor()+255)/545, self.ball.dx/580, self.ball.dy/545]

    def step(self, action, render=True):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()

        if action == 2:
            self.paddle_right()

        self.run_frame()

        state = [(self.paddle.xcor()+290)/580, (self.ball.xcor()+290)/580,
                 (self.ball.ycor()+255)/545, self.ball.dx/580, self.ball.dy/545]
        return state, self.reward, self.done, {}

    def run_frame(self):

        self.win.update()

        # 球移动

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # 碰撞检查

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # 球落下了

        if self.ball.ycor() < -290:
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(
                self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # 球和浆的碰撞

        if self.ball.ycor() + 255 >= 0 and self.ball.ycor() + 255 < -self.ball.dy and abs(self.paddle.xcor() - self.ball.xcor()) <= 60:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(
                self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3
            if self.hit == 50:
                self.done = True



if __name__ == '__main__':
    env = Paddle()
    import time

    while True:
        env.run_frame()
        time.sleep(0.01)
        if env.done:
            print(env.reward)
            env.reset()
            env.done = False
            env.reward = 0
