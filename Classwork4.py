"""
Subject : Artificial Intelligence
Assignment Code: Classwork #4
StudID: 6410301026
StudName: Thanabodin Keawmaha
Deptment: CPE
Due Date: 2024-2-6
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras import layers

def generateData(data_pattern):
    df_data = pd.DataFrame()
    if data_pattern=='blobs':
        X_train, y_train = make_blobs(n_samples=100,
                              n_features=2,
                              centers=2,
                              cluster_std=0.2,
                              center_box=(0,5))
    elif data_pattern=='circles':
        X_train, y_train = make_circles(n_samples=100,
                              noise=0.1,
                              factor=0.2)
    elif data_pattern=='moons':
        X_train, y_train = make_moons(n_samples=100,
                              noise=.05)
    df_data['x'] = X_train[:,0]
    df_data['y'] = X_train[:,1]
    df_data['cluster'] = y_train
    return df_data

def nomalize_data(x):
    ss = StandardScaler()
    X_ss = ss.fit_transform(x)
    return X_ss

def plot_decision_boundary(model, X, Y):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(y_min, y_max, h), np.arange(x_min, x_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)

    cmap_points = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_points, edgecolors='k', marker='o')

    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def blobs_model(data):
    X = data[['x', 'y']].values
    y = data['cluster'].values

    #nomalize data
    X = nomalize_data(X)

    #split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    input_shape = X_train.shape[1]

    #make model
    model = Sequential([
    layers.Input(shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #train model data
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'loss: {score[0]}')
    print(f'accuracy: {score[1]}')

    prediction = model.predict(X_test)
    y_pred = np.where(prediction>0.5, 1, 0)
    print(y_pred[:5])

    #plot data
    plot_decision_boundary(model,X_train,y_train)
    
def circles_model(data):
    X = data[['x', 'y']].values
    y = data['cluster'].values

    #nomalize data
    X = nomalize_data(X)

    #split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    input_shape = X_train.shape[1]

    #make model
    model = Sequential([
    layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #train model data
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'loss: {score[0]}')
    print(f'accuracy: {score[1]}')

    prediction = model.predict(X_test)
    y_pred = np.where(prediction>0.5, 1, 0)
    print(y_pred[:5])

    #plot data
    plot_decision_boundary(model,X_train,y_train)    

def moons_model(data):
    X = data[['x', 'y']].values
    y = data['cluster'].values

    #nomalize data
    X = nomalize_data(X)

    #split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    input_shape = X_train.shape[1]

    #make model
    model = Sequential([
    layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(96, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #train model data
    model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'loss: {score[0]}')
    print(f'accuracy: {score[1]}')

    prediction = model.predict(X_test)
    y_pred = np.where(prediction>0.5, 1, 0)
    print(y_pred[:5])

    #plot data
    plot_decision_boundary(model,X_train,y_train)   

if __name__ == "__main__":

    np.random.seed(42)
    
    # Blobs dataset Model
    data_blobs = generateData('blobs')
    blobs_model(data_blobs)

    # Blobs dataset Model
    # data_circles = generateData('circles')
    # circles_model(data_circles)

    # moons dataset Model
    # data_moons = generateData('moons')
    # moons_model(data_moons)

    print(data_blobs)