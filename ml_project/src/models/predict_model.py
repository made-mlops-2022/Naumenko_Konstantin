from operator import mod
import numpy as np
import logging
import pickle


def load_model(path):
    with open(path, 'wb') as f:
        model = pickle.load(f)
    return model

def predict(data):
    model = load_model('path')
    res = model.predict(data)
    return res


if __name__ == '__main__':
    print(predict())
