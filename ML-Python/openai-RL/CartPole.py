import gym, random, tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
environment = gym.make('CartPole-v0')#choose the environmentironment
environment.reset()#start the environmentironment
frames = 300
to_win = 50
to_play = 10000

def starter():
    to_train = []
    scores = []
    score_to_save = []
    for _ in range(to_play):
        score = 0
        played = []
        prev_obs = []
        for _ in range(frames):
            action = random.randrange(0,2)
            observation, reward, done, info = environment.step(action)
            if len(prev_obs) > 0:
                played.append([prev_obs, action])
            prev_obs = observation
            score += reward
            if done:
                break
        if score >= to_win:
            score_to_save.append(score)
            for data in played:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                to_train.append([data[0],output])
        environment.reset()
        scores.append(score)
    #to_train_save = np.array(to_train)
    #np.save('saved.npy', to_train_save)
    print('Average accepted score: ', mean(score_to_save))
    print('Median accepted score: ', median(score_to_save))
    print(Counter(score_to_save))
    return to_train


def NNModel(input_size):
    layer = input_data(shape=[None, input_size,1], name='input')
    layer = fully_connected(layer, 128, activation='crelu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 256, activation='crelu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 512, activation='crelu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 256, activation='crelu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 128, activation='crelu')
    layer = dropout(layer, 0.8)

    layer = fully_connected(layer, 2, activation='sigmoid')

    layer = regression(layer, optimizer='adam', learning_rate=LR,
                            loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(layer, tensorboard_dir='log')
    return model

def train_model(train_data, model=False):
    X = np.array([i[0] for i in to_train]).reshape(-1, len(to_train[0][0]),1)
    y = [i[1] for i in to_train]
    if not model:
        model = NNModel(input_size = len(X[0]))
    model.fit({'input': X},{'targets':y}, n_epoch=3, snapshot_step=500, show_metric=True,
                run_id='gym.openAI')
    return model
to_train = starter()
model = train_model(to_train)

def run_game():
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        played = []
        prev_obs = []
        environment.reset()
        for _ in range(frames):
            environment.render()
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
            choices.append(action)
            new_obs, reward, done, info = environment.step(action)
            prev_obs = new_obs
            played.append([new_obs, action])
            score += reward
            if done:
                break
        scores.append(score)
    print('Average Score', sum(scores)/len(scores))
    print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices),
            choices.count(0)/len(choices)))
run_game()
