import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 0)

df = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

def create_plot(subset=df, x='Age', y='Fare'):
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

def show_survival_rate(subset=df, desc=''):
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

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

print(df.head(20))
# create_plot()
# plt.show()

# TODO: analyze data: what can be dropped?
# name, ticket

# TODO: split data in training/validation set (random)



# TODO: define neural network

# TODO: run network, evaluate

#
