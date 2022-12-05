#!/usr/bin/env python3

from mpitree.random_forest import *

import time
import argparse
from os.path import exists
from pickle import load, dump
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, required=True, help="specifies the number of decision trees")
args = parser.parse_args()



"""Predicting Anime Rating using Random Forest"""



if __name__ == "__main__":
    model = fr'models/anime_{args.n}_sequential.sav'

    df = load(open("df_anime_x.p", "rb"))
    df = df[["Demographic", "Source Material", "Rating"]]

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    start_time = time.time()

    if exists(model):
        print("Loading", model)
        rf = load(open(model, 'rb'))
    else:
        rf = RandomForest(n_estimators=args.n, n_sample=len(X_train))
        rf.fit(X_train, y_train)
        dump(rf, open(model, 'wb'))

    print(rf)
    score = rf.score(X_test, y_test)

    print(f"MSE Score: {score:.2f}")
    print(f"Sequential Execution Time: {time.time() - start_time:.3f}s")
