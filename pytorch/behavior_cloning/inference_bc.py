import torch
from torch.distributions import Categorical

import my_env # import your env
import bc


def train():
    env = my_env.MyEnv(0, realtime_mode=True)
    model = bc.BC()
    model.load_state_dict(torch.load("imitation_model_1000.pt"))
    score = 0.0
    print_interval = 1

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            prob = model(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()

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

