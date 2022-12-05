#!/usr/bin/env python3

from mpi4py import MPI
from mpitree.parallel_random_forest import *

from os.path import exists
from pickle import load, dump
from sklearn.model_selection import train_test_split



"""Predicting Anime Rating using Distributed Random Forest"""



if __name__ == "__main__":
    model = fr'models/anime_{size}_parallel.sav'

    df = load(open("df_anime_x.p", "rb"))
    df = df[["Demographic", "Source Material", "Rating"]]

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    comm.Barrier()
    start_time = MPI.Wtime()

    if exists(model):
        if not rank:
            print("Loading", model)
        forest = load(open(model, 'rb'))
        dt = forest[rank]
    else:
        dt = RandomForest(n_sample=len(X_train))
        dt.fit(X_train, y_train)
        forest = comm.gather(dt, root=0)

    print(rank, dt.tree)
    score = dt.score(X_test, y_test)

    end_time = MPI.Wtime()
    if not rank:
        print(f"MSE Score: {score:.2f}")
        print(f"Parallel Execution Time: {end_time - start_time:.3f}s")
        dump(forest, open(model, 'wb'))
