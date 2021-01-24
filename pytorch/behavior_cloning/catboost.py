import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import Pool, CatBoostClassifier

np_file = np.load("food.npz")

states = np_file["states"]
actions = np_file["actions"]
print(np.shape(states))
print(np.shape(actions))

kfold = StratifiedKFold(n_splits=5)
for train_index, test_index in kfold.split(states, actions):
    print(len(test_index))

x_train, x_test = states[train_index, :], states[test_index, :]
y_train, y_test = actions[train_index], actions[test_index]


train_dataset = Pool(data=x_train, label=y_train)
eval_dataset = Pool(data=x_test, label=y_test)

params = {
      'depth': 8,
      'learning_rate': 0.1,
      'random_seed': 42,
      'iterations': 1000,
      'task_type': 'GPU',
      "use_best_model": True,
}

model = CatBoostClassifier(**params)
model.fit(train_dataset, eval_set=eval_dataset)

model.save_model("catboost_model.model")