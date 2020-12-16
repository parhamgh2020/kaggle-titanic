from pandas import read_csv, concat, get_dummies, DataFrame

from sklearn.model_selection import train_test_split as split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator

train = read_csv('./data/train.csv')
test = read_csv('./data/test.csv')

PassengerId = train.PassengerId.to_numpy()
y_trian = train.Survived

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
# print(data.info())
data = IterativeImputer().fit_transform(data)
# data = KNNImputer().fit_transform(data)
# data = MissingIndicator().fit_transform(data)
# print(data.shape)
x_train, x_test = data[:len(train)], data[len(train):]
x_train = scale(x_train)
m1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=len(x_train[0]) * 2)

cv = cross_val_score(m1, x_train, y_trian, cv=5)
print(cv.mean(), ' +/-', cv.std() * 2)

x_tr, x_ts, y_tr, y_ts = split(x_train, y_trian, shuffle=True, test_size=0.2)
m1.fit(x_tr,y_tr)
print(m1.score(x_ts, y_ts))

prediction = m1.predict(x_test)
df = DataFrame(data=list(zip(PassengerId, prediction)), columns=['PassengerId', 'Survived'])
df.to_csv('submission.csv', index=False)
