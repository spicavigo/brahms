import os

import json
import csv
from math import pi

import argparse

from scipy import misc

from keras.optimizers import SGD
from keras.models import model_from_json

from simulator import simulate

def gen(dataset):
    '''
    CSV format:
    Latitude, Longitude, Gear, Brake, Throttle, Steering Angle, Speed, CameraEnum, FileName
    where CameraEnum is one of L, C, R
    '''
    with open(dataset, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for row in reader:
            if row[-2].strip() != 'C':
                continue
            steering = float(row[-4])
            speed = float(row[-3])
            if speed < 7.: continue

            img = row[-1].strip()
            if not os.path.exists(img): continue

            ts = int(img.split('/')[1].split('.')[0])
            image_disp = misc.imread(img)
            image_pred = image_disp[320:, :, :]
            yield image_pred, image_disp, speed, steering, ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN viewer')
    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--dataset', type=str, help='Driving log file', required=True)
    args = parser.parse_args()

    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(sgd, "mse")
    weights_file = args.model.replace('json', 'keras')
    model.load_weights(weights_file)

    dataset = args.dataset
    prediction_fn = lambda image_pred: model.predict(image_pred[None, :, :, :])[0][0]
    print simulate(prediction_fn, gen(dataset), 640, 480, error_limit=pi/6)
