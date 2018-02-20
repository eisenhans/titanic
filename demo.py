import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 0)

main_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def create_plot(subset=main_df, x='Age', y='Fare'):
    markers = {0: 'x', 1: 'o'}
    colors = {'female': 'red', 'male': 'blue'}
    labels = {(0, 'female'): 'female deceased', (1, 'female'): 'female survived',
              (0, 'male'): 'male deceased', (1, 'male'): 'male survived'}

    fig, ax = plt.subplots()
    # ax.set_ylim([-3, 100])
    grouped = subset.groupby(['Survived', 'Sex'])

    for key, group in grouped:
        group.plot(ax=ax, x=x, y=y, kind='scatter', marker=markers[key[0]], color=colors[key[1]], label=labels[key])


# babies = df.query('Age < 10.0')
# create_plot(babies)
# plt.show()

def show_survival_rate(subset=main_df, desc=''):
    survived = len(subset[subset['Survived'] == 1])
    total = len(subset)
    print('{}: survival rate {:.1f}% ({}/{})'.format(desc, survived/total*100, survived, total))


def first_char(x):
    if not isinstance(x, str):
        return None
    return x[0]


def starts_with_letter(x):
    return first_char(x).isalpha()


# show_survival_rate(df, 'Total')
# for key, group in df.groupby('Sex'):
#     show_survival_rate(group, key)

def prepare_data(df):
    df['Age_lt2'] = np.where(df['Age'] < 2, 1, 0)
    df['Age_lt5'] = np.where((df['Age'] >= 2) & (df['Age'] < 5), 1, 0)
    df['Age_lt10'] = np.where((df['Age'] >= 5) & (df['Age'] < 10), 1, 0)
    df['Age_lt20'] = np.where((df['Age'] >= 10) & (df['Age'] < 20), 1, 0)
    df['Age_lt40'] = np.where((df['Age'] >= 20) & (df['Age'] < 40), 1, 0)
    df['Age_lt60'] = np.where((df['Age'] >= 40) & (df['Age'] < 60), 1, 0)
    df['Age_lt100'] = np.where(df['Age'] >= 60, 1, 0)

    # replace cabin with first char
    df['Cabin'] = df['Cabin'].apply(first_char)

    df['male'] = np.where(df['Sex'] == 'male', 1, 0)
    cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin', drop_first=False)
    df = pd.concat([df, cabin_dummies], axis=1)

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=False)
    df = pd.concat([df, embarked_dummies], axis=1)

    # standardize fare
    max_fare = df.Fare.max()
    df.Fare = df.Fare / max_fare

    df = df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    return df


main_df = prepare_data(main_df)
print(main_df.head(20))
# create_plot()
# plt.show()

# split data in training/validation set (random)
# TODO: better use cross-validation
temp_df, test_local_df = train_test_split(main_df, test_size=0.2)
train_df, val_df = train_test_split(temp_df, test_size=0.25)
print('train/validate/test/total data: {}/{}/{}/{}'.format(len(train_df), len(val_df), len(test_df), len(main_df)))

target_col = 'Survived'
x_train = train_df.drop(target_col, axis=1).values
y_train = train_df[target_col].values

x_val = val_df.drop(target_col, axis=1).values
y_val = val_df[target_col].values
x_local_test = test_local_df.drop(target_col, axis=1).values
y_local_test = test_local_df[target_col].values

# define neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
weights_file = 'output/weights_best_model.hdf5'

checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)

# train model
hist = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val), callbacks=[checkpointer],
                 verbose=2)

model.load_weights(weights_file)

train_score = model.evaluate(x_train, y_train, verbose=0)
val_score = model.evaluate(x_val, y_val, verbose=0)
test_score = model.evaluate(x_local_test, y_local_test, verbose=0)
print("Training/validation/test accuracy: ", train_score[1], val_score[1], test_score[1])

# TODO: create output for upload
test_passenger_ids = test_df.PassengerId.values
test_passenger_ids.shape = (test_passenger_ids.shape[0], 1)
test_df = prepare_data(test_df)
if not 'Cabin_T' in test_df:
    test_df['Cabin_T'] = 0
x_test = test_df.values
prediction = model.predict(x_test)
pred_int = (prediction >= 0.5).astype(int)


result = np.concatenate((test_passenger_ids, pred_int), axis=1)
np.savetxt("output/result.csv", result, delimiter=",", fmt='%1i', header='PassengerId,Survived', comments='')

# TODO: visualize: what whas right/wrong?