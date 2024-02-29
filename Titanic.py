import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
from keras.layers import Dense
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.regularizers import l1, l2


def nomalize_data(x):
    ss = StandardScaler()
    X_ss = ss.fit_transform(x)
    return X_ss

def conver_type_data(data,col):
    x = []
    for i in data:
        if i == col:
           x.append(1)
        else:
           x.append(0)
    return x

if __name__ == "__main__":

    url = "titanic.csv"
    data = pd.read_csv(url)

    # X = data[['Age']].values
    # x1 = []
    # for i in X:
    #     if i[np.isnan(i)]:
    #        x1.append(0)
    #     else:
    #        x1.append(1)
    # data['check'] = x1

    bin_sex = conver_type_data(data["Sex"].values, "male")
    data["bin_sex"] = bin_sex

    X = data[['bin_sex', 'Fare', 'Pclass']].values
    y = data['Survived'].values

    X[np.isnan(X)] = np.nanmean(X)

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

    # # Initialising the NN
    # model = Sequential()

    # # layers
    # model.add(Dense(32, kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=l1(0.01), input_dim =  input_shape))
    # model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=l2(0.01)))
    # # model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    # model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # # summary
    # model.summary()
    # model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #train model data
    model.fit(X_train, y_train, epochs=200, batch_size=25, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'loss: {score[0]}')
    print(f'accuracy: {score[1]}')

    prediction = model.predict(X_test)
    y_pred = np.where(prediction>0.5, 1, 0)
    print(y_pred[:5])

     # คำนวณ ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # คำนวณ ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC:", roc_auc)

    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')

    # เพิ่มค่า ROC AUC ลงในกราฟ
    plt.text(0.6, 0.2, f'ROC AUC = {roc_auc:.2f}', fontsize=12)

    plt.show()
    # plot_decision(model,X_train,y_train)   

