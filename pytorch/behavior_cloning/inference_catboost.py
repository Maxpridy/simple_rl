import numpy as np
from catboost import Pool, CatBoostClassifier

import my_env

deterministic = True

def action_mapping(max_value):
    if max_value == 0:
        return 4
    elif max_value == 1:
        return 1
    else:
        return max_value+1

def train():
    env = my_env.MyEnv(0, realtime_mode=True)

    model = CatBoostClassifier()
    model.load_model("catboost_model.model")

    score = 0.0
    print_interval = 1

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            y_pred1 = model.predict(s, prediction_type="Probability")
            
            if deterministic:
                y_pred_max = int(np.argmax(y_pred1))
                a = action_mapping(y_pred_max)
            else:
                a = int(np.random.choice([0, 1, 3, 4, 5], p=y_pred1))            
            
            s_prime, r, done, info = env.step(a)

            s = s_prime

            score += r
            if done:
                break

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.5f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()


if __name__ == "__main__":
    train()

