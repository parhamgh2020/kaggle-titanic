from pandas import read_csv, concat, get_dummies, DataFrame
from sklearn.model_selection import train_test_split as split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator
import matplotlib.pyplot as plt

train = read_csv('/home/p/Desktop/kagglee/titanic/kaggle-titanic/data/train.csv')
test = read_csv('/home/p/Desktop/kagglee/titanic/kaggle-titanic/data/test.csv')

PassengerId = test.PassengerId.to_numpy()
y_train = train.Survived

x_train = train.drop(['PassengerId', 'Survived'], axis=1)
x_test = test.drop(['PassengerId'], axis=1)

data = concat([x_train, x_test], axis=0)


def make_one_hot(data, field):
    temp = get_dummies(data[field], prefix=field)
    data.drop(field, inplace=True, axis=1)
    return concat([data, temp], axis=1)


def adjust_title(title):
    if title in ['Mr', 'Miss', 'Mrs', 'Ms']:
        return 'English'
    elif title in ['Sir', 'Lady', 'the Countess']:
        return 'EnglishNoble'
    elif title in ['Major', 'Col', 'Capt']:
        return 'Military'
    elif title in ['Mlle', 'Mme', 'Don', 'Dona', 'Jonkheer']:
        return 'OtherEuropean'
    else:
        return 'SomeImportantGuy'  # Dr, Religious, Master


def feature_eng(data):
    # objects null fixing
    data['Cabin'].fillna('*', inplace=True)
    data['Embarked'].fillna('*', inplace=True)
    # make object ready for one_hot operation
    data.Name = data.Name.map(
        lambda n: n.split(',')[1].split('.')[0].strip())
    data.Cabin = data.Cabin.map(lambda a: a[0])
    # one_hot
    for i in ['Cabin', 'Embarked', 'Name']:
        data = make_one_hot(data, i)
    data.Sex.replace({'female': 0, 'male': 1}, inplace=True)
    # numeric feature eng
    data['Family_size'] = data.Parch + data.SibSp + 1
    # drop
    data.drop(['Cabin_*', 'SibSp', 'Parch', 'Ticket'], axis=1, inplace=True)

    return data


data = feature_eng(data)
data = IterativeImputer().fit_transform(data)
x_train, x_test = data[:len(train)], data[len(train):]
x_train = scale(x_train)
m1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=len(x_train[0]) * 2)

cv = cross_val_score(m1, x_train, y_train, cv=5)
print(cv.mean(), ' +/-', cv.std() * 2)

x_tr, x_ts, y_tr, y_ts = split(x_train, y_train, shuffle=True, test_size=0.2)
m1.fit(x_tr, y_tr)
print(m1.score(x_ts, y_ts))

# ===================================
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.activations import sigmoid, relu
from keras.losses import BinaryCrossentropy
from numpy import argmax


def plot_history(net_history):
    from matplotlib.pyplot import clf
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']

    clf()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])

    # plt.figure()
    # clf()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.plot(accuracies)
    # plt.plot(val_accuracies)
    # plt.legend(['accuracy', 'val_accuracy'])


def loss_frame(loss_test, his):
    loss_train = his.history['loss'][-1]
    loss_val = his.history['val_loss'][-1]
    df = DataFrame([loss_train, loss_val, loss_test], index=['train', 'val', 'test'], columns=['loss'])
    print(df)


rows, cols = x_train.shape
x_tr, x_ts, y_tr, y_ts = split(x_train, y_train, shuffle=True, test_size=0.1)

# ========= train , val =================
model = Sequential()
model.add(Dense(1, input_dim=cols, activation='sigmoid', kernel_regularizer=l2()))
model.compile(loss=BinaryCrossentropy(), metrics='accuracy')
his = model.fit(x_tr, y_tr, validation_split=0.1, epochs=25, verbose=1)
plot_history(his)
# =========== comparison  train , val , test ===========
his2 = model.fit(x_ts, y_ts, epochs=25)
loss_frame(his2.history['loss'][-1], his)

# ============== submission =========================
model.fit(x_train, y_train, epochs=35, verbose=0)
predict = model.predict(x_test)
predict = predict.reshape(-1)
predict[predict >= 0.5] = 1
predict[predict < 0.5] = 0
submission = DataFrame(list(zip(PassengerId, predict)), columns=['PassengerId', 'Survived'],dtype=int)
submission.to_csv('submission.csv', index=False)
