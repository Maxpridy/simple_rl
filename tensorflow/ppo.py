# I referred to 
# https://github.com/seungeunrho/minimalRL
# https://github.com/tensorflow/agents/blob/e72484a13ed2288d49066139d42c5d120a7dc7f7/tf_agents/utils/value_ops.py#L98
# but transpose and GAE take too long. Need improvement.

import gym

import numpy as np
import tensorflow as tf

learning_rate = 0.0005
gamma = 0.98
td_lambda = 0.95
eps_clip = 0.1
train_epoch = 3
T_horizon = 100

class PPO(tf.keras.Model):

    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1 = tf.keras.layers.Dense(64, input_shape=(state_dim,), activation=tf.nn.swish, kernel_initializer='orthogonal')
        self.fc2 = tf.keras.layers.Dense(64, activation=tf.nn.swish, kernel_initializer='orthogonal')
        self.actor = tf.keras.layers.Dense(action_dim, kernel_initializer='orthogonal')
        self.critic = tf.keras.layers.Dense(1, kernel_initializer='orthogonal')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def pi(self, state):
        fc1 = self.fc1(state)
        fc2 = self.fc2(fc1)
        logits = self.actor(fc2)
        probs = tf.nn.softmax(logits)
        
        return probs, logits

    def v(self, state):
        fc1 = self.fc1(state)
        fc2 = self.fc2(fc1)
        v = self.critic(fc2)
        return v

    def store_data(self, input_data):
        self.data.append(input_data)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s = np.array(s_lst)
        a = tf.convert_to_tensor(a_lst)
        r = tf.convert_to_tensor(r_lst)
        s_prime = np.array(s_prime_lst)
        done_mask = tf.convert_to_tensor(done_lst, dtype=tf.float32)
        prob_a = tf.convert_to_tensor(prob_a_lst)
        
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()


        for i in range(train_epoch):
            with tf.GradientTape() as tape:
                values = self.v(s)
                next_values = self.v(s_prime)

                td_target = r + gamma * next_values * done_mask

                constant_gamma = tf.constant([[gamma]] * np.shape(values)[0])
                
                delta = r + constant_gamma * next_values - values
                weighted_discounts = constant_gamma * td_lambda

                delta = tf.transpose(delta)
                weighted_discounts = tf.transpose(weighted_discounts)

                def weighted_cumulative_td_fn(accumulated_td, reversed_weights_td_tuple):
                    weighted_discount, td = reversed_weights_td_tuple
                    return td + weighted_discount * accumulated_td

                advantage = tf.nest.map_structure(tf.stop_gradient, tf.scan(fn=weighted_cumulative_td_fn, elems=(weighted_discounts, delta), initializer=tf.zeros_like(delta), reverse=True))
                advantage = tf.reshape(advantage, (-1, 1))

                pi, _ = self.pi(s)
                pi_a = tf.gather_nd(pi, a, 1)
                pi_a = tf.expand_dims(pi_a, 1)

                ratio = tf.math.exp(tf.math.log(pi_a) - tf.math.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1-eps_clip, 1+eps_clip) * advantage
                
                huber_loss = tf.keras.losses.Huber()
                loss = -tf.math.minimum(surr1, surr2) + huber_loss(self.v(s), tf.stop_gradient(td_target))

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            
            del tape


def main():
    env = gym.make('CartPole-v1')
    model = PPO(4, 2)
    score = 0.0
    print_interval = 20
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                probs, logits = model.pi(tf.reshape(tf.convert_to_tensor(s), (1, -1)))
                a = tf.random.categorical(logits, 1)[0, 0]

                s_prime, r, done, info = env.step(a.numpy())

                model.store_data((s, a, r/100.0, s_prime, probs.numpy()[0][a], done))
                s = s_prime

                score += r
                if done:
                    break
                
            model.train_net()


        if n_epi%print_interval==0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()