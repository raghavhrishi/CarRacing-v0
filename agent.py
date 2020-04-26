import numpy as np
from skimage import color, transform
import tensorflow.contrib.slim as slim
import tensorflow as tf
import itertools as it
import tensorflow.contrib as tf_contrib
from dqn.experience_history import ExperienceHistory
import sys 
from scipy.special import rel_entr
from keras.backend import clear_session
from tensorflow.keras import datasets, layers, models
import math     
from PIL import Image

class DQN:
    """
    General DQN agent.
    Can be applied to any standard environment
    The implementation follows:
    Mnih et. al - Playing Atari with Deep Reinforcement Learning https://arxiv.org/pdf/1312.5602.pdf
    The q-network structure is different from the original paper
    see also:
    David Silver's RL course lecture 6: https://www.youtube.com/watch?v=UoPei5o4fps&t=1s
    """

    def __init__(self,
            env,
            batchsize=64,
            pic_size=(96, 96),
            num_frame_stack=4,
            gamma=0.95,
            frame_skip=1,
            train_freq=4,
            initial_epsilon=1.0,
            min_epsilon=0.1,
            render=True,
            epsilon_decay_steps=int(1e6),
            min_experience_size=int(1e3),
            experience_capacity=int(1e5),
            network_update_freq=5000,
            regularization = 1e-6,
            optimizer_params = None,
            action_map=None
    ):
        self.exp_history = ExperienceHistory(
            num_frame_stack,
            capacity=experience_capacity,
            pic_size=pic_size
        )

        # in playing mode we don't store the experience to agent history
        # but this cache is still needed to get the current frame stack
        self.playing_cache = ExperienceHistory(
            num_frame_stack,
            capacity=num_frame_stack * 5 + 10,
            pic_size=pic_size
        )

        if action_map is not None:
            self.dim_actions = len(action_map)
        else:
            self.dim_actions = env.action_space.n

        self.network_update_freq = network_update_freq
        self.action_map = action_map
        self.env = env
        self.batchsize = batchsize
        self.num_frame_stack = num_frame_stack
        self.gamma = gamma
        self.frame_skip = frame_skip
        self.train_freq = train_freq
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.render = render
        self.min_experience_size = min_experience_size
        self.pic_size = pic_size
        self.regularization = regularization
        # These default magic values always work with Adam
        self.optimizer_params = optimizer_params or dict(learning_rate=0.0004, epsilon=1e-7)

        self.do_training = True
        self.playing_epsilon = 0.0
        self.session = None

        self.state_size = (self.num_frame_stack,) + self.pic_size
        self.global_counter = 0
        self.episode_counter = 0

    @staticmethod
    def process_image(img):
        return 2 * color.rgb2gray(transform.rescale(img[34:194], 0.5)) - 1



    def kl_divergence(p, q): 
        return tf.reduce_sum(p * tf.log(p/q))


    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def MakeItWork(self, x):
        
        x = tf.placeholder(tf.float32, [64, 96, 96, 3], name='image')
        x = tf.image.resize_images(x, [64, 64])
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        red_work = x
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        # x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        # x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        # x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.flatten(x)
        x = tf.reshape(x, [-1, 6272])
        print(x)
        z_mu = slim.fully_connected(x, 5, activation_fn=tf.nn.elu)
        z_var = slim.fully_connected(x, 5, activation_fn=tf.nn.elu)
        print("slim",z_mu)
        return z_mu, z_var

    def MakeItWorkEvent(self, x):
        
        x = tf.placeholder(tf.float32, [64, 96, 96, 1], name='image')
        x = tf.image.resize_images(x, [64, 64])
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        red_work = x
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        # x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        # x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        # x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.flatten(x)
        x = tf.reshape(x, [-1, 6272])
       
        print(x)
        z_mu = slim.fully_connected(x, 5, activation_fn=tf.nn.elu)
        z_var = slim.fully_connected(x, 5, activation_fn=tf.nn.elu)
        print("slim",z_mu)

        return z_mu, z_var




    def encoderRGB(self, x):
        
        x = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        x = tf.image.resize_images(x, [96, 96])
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.flatten(x)
        fc1 = tf.reshape(x, [-1, 4096])
        shapes = tf.shape(x)
        #z_mu = slim.fully_connected(fc1, 5, activation_fn=tf.nn.elu)
        z_mean = tf_contrib.layers.fully_connected(fc1,32)

        shape = x.get_shape().as_list() 
        z_mua = tf.layers.dense(fc1, units=32, name='z_mu')
        z_logvara = tf.layers.dense(fc1, units=32, name='z_logvar')
        # dim = np.prod(shape[1:])            
        # x2 = tf.reshape(-1, x.get_shape())
        #print("dimension!!!",fc1)
       
               # x = tf.reshape(-1,4096)
        tf.reset_default_graph()

        # z_mus = tf.layers.dense(x2, units=32, name='z_mu')
        # z_logvars = tf.layers.dense(x2, units=32, name='z_logvar')

        return z_mean, z_logvara

    
    def encoderEvent(self, x):
        #x = tf.placeholder(tf.float32, [None, None, 96, 96], name='image')

        model = models.Sequential()
        model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=[None, 96, 96]))
        model.add(layers.MaxPooling2D((2, 2)))

        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        x = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        shape = x
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.elu)
        x = tf.layers.flatten(x)
        fc1 = tf.reshape(x, [-1, 4096])
        z_means = tf.layers.dense(fc1, units=32, name='z_mu')
        z_logvariance = tf.layers.dense(fc1, units=32, name='z_logvar')
       # print("dimension!!!!!!!!!!1",type(x))
        # x = tf.reshape(x.size(0),-1)
        # z_mu = tf.layers.dense(x, units=32, name='z_mu')
        # z_logvar = tf.layers.dense(x, units=32, name='z_logvar')
        tf.reset_default_graph()
        return z_means, z_logvariance, x

    def compute_loss(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_image)
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return vae_loss 

    def build_graph(self):
        

        input_dim_with_batch = (self.batchsize, self.num_frame_stack) + self.pic_size
        input_dim_general = (None, self.num_frame_stack) + self.pic_size

        self.input_prev_state = tf.placeholder(tf.float32, input_dim_general, "prev_state")
        self.input_next_state = tf.placeholder(tf.float32, input_dim_with_batch, "next_state")
        self.input_reward = tf.placeholder(tf.float32, self.batchsize, "reward")
        self.input_actions = tf.placeholder(tf.int32, self.batchsize, "actions")
        self.input_done_mask = tf.placeholder(tf.int32, self.batchsize, "done_mask")

        # These are the state action values for all states
        # The target Q-values come from the fixed network
        with tf.variable_scope("fixed"):
            qsa_targets = self.create_network(self.input_next_state, trainable=False)

        with tf.variable_scope("train"):
            qsa_estimates = self.create_network(self.input_prev_state, trainable=True)

        self.best_action = tf.argmax(qsa_estimates, axis=1)

        not_done = tf.cast(tf.logical_not(tf.cast(self.input_done_mask, "bool")), "float32")
        q_target = tf.reduce_max(qsa_targets, -1) * self.gamma * not_done + self.input_reward
        # select the chosen action from each row
        # in numpy this is qsa_estimates[range(batchsize), self.input_actions]
        action_slice = tf.stack([tf.range(0, self.batchsize), self.input_actions], axis=1)
        q_estimates_for_input_action = tf.gather_nd(qsa_estimates, action_slice)

        training_loss = tf.nn.l2_loss(q_target - q_estimates_for_input_action) / self.batchsize

        optimizer = tf.train.AdamOptimizer(**(self.optimizer_params))

        reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.train_op = optimizer.minimize(reg_loss + training_loss)

        train_params = self.get_variables("train")
        fixed_params = self.get_variables("fixed")

        assert (len(train_params) == len(fixed_params))
        self.copy_network_ops = [tf.assign(fixed_v, train_v)
            for train_v, fixed_v in zip(train_params, fixed_params)]

    def get_variables(self, scope):
        vars = [t for t in tf.global_variables()
            if "%s/" % scope in t.name and "Adam" not in t.name]
        return sorted(vars, key=lambda v: v.name)

    def create_network(self, input, trainable):
        if trainable:
            wr = slim.l2_regularizer(self.regularization)
        else:
            wr = None

        # the input is stack of black and white frames.
        # put the stack in the place of channel (last in tf)
        input_t = tf.transpose(input, [0, 2, 3, 1])
        net = slim.conv2d(input_t, 8, (7, 7), data_format="NHWC",
            activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.conv2d(net, 16, (3, 3), data_format="NHWC",
            activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
            weights_regularizer=wr, trainable=trainable)
        q_state_action_values = slim.fully_connected(net, self.dim_actions,
            activation_fn=None, weights_regularizer=wr, trainable=trainable)

        return q_state_action_values

    def check_early_stop(self, reward, totalreward):
        return False, 0.0

    def get_random_action(self):
        return np.random.choice(self.dim_actions)

    def get_epsilon(self):
        if not self.do_training:
            return self.playing_epsilon
        elif self.global_counter >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            # linear decay
            r = 1.0 - self.global_counter / float(self.epsilon_decay_steps)
            return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * r

    def train(self):
        batch = self.exp_history.sample_mini_batch(self.batchsize)

        fd = {
            self.input_reward: "reward",
            self.input_prev_state: "prev_state",
            self.input_next_state: "next_state",
            self.input_actions: "actions",
            self.input_done_mask: "done_mask"
        }
        fd1 = {ph: batch[k] for ph, k in fd.items()}
        self.session.run([self.train_op], fd1)
    def kl(self,x, y):
        X = tf.distributions.Categorical(probs=x)
        Y = tf.distributions.Categorical(probs=y)
        return tf.distributions.kl_divergence(X, Y)



    def play_episode(self):
        eh = (
            self.exp_history if self.do_training
            else self.playing_cache
        )
        total_reward = 0
        frames_in_episode = 0
        with tf.compat.v1.Session() as sess:

            first_frame = self.env.reset()
            val, event = self.env.returnRgb()

            img_val = Image.fromarray(val)
            img_event = Image.fromarray(event)
            
            event_shape = tf.shape(event)
            k = tf.keras.losses.KLDivergence()
            vals_mu, vals_var = self.MakeItWork(val)
            events_mu, events_var = self.MakeItWorkEvent(event)
            #Sampling into a latent vector
            val_latent = self.sample_z(vals_mu,vals_var)
            val_event = self.sample_z(events_mu,events_var)
            #kl_loss = 0.5 * tf.exp(events_var) 
            kl_loss = 0.5 * tf.reduce_sum(tf.exp(events_var) + events_mu**2 - 1. - events_var, 1)
            #KL Divergence
            X = tf.distributions.Categorical(probs=val_latent)
            Y = tf.distributions.Categorical(probs=val_event)
            flat_vector = tf.distributions.kl_divergence(X, Y)
            print("The ")
            # mu, var = self.encoderRGB(val)
            # event_mu, event_logvar, xu = self.encoderEvent(event)
            #latent = self.sample_z(mu,var)
            #latent_event = self.sample_z(event_mu,event_logvar)
            #print(xu.shape)
            # loss = val_latent * math.log(val_latent / val_event)


            value = tf.math.add(val_latent,val_event)
            #value = k(val_latent,val_event)
            print("KL", flat_vector)
            #result = sess.run(val_latent)
            # sess = tf.InteractiveSession()
            # xo = tf.Print(latent,[latent])

            # sess.close()
        #value = tfp.distributions.kl_divergence(latent, latent_event, allow_nan_stats=True, name=None)

        #kl_r = rel_entr(latent,latent_event)
        #clear_session()
        # a = tf.constant([[4,3],[3,3]])
        # print(type(a))
        # sess = tf.InteractiveSession()
        # xo = tf.Print(mu,[mu])
        # sess.run(xo)
        # sess.close()
        wr = slim.l2_regularizer(self.regularization)

        # k = tf.keras.losses.KLDivergence()
        # loss = k(latent,latent_event)

        #PRINTING VALUES
       #  a = tf.constant([[4,3],[3,3]])
       # # x = tf.Print(a,[a])
       #  sess = tf.InteractiveSession()
       #  sess.run(latent_event)
       #  sess.close()



        # net = slim.conv2d(event, 8, (7, 7), data_format="NHWC",
        #     activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr)

        # x = tf.layers.conv2d(event, filters=32, kernel_size=1,  activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2,  activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2,  activation=tf.nn.relu)
        # # x = tf.layers.flatten(x)
        # z_mu = tf.layers.dense(x, units=32, name='z_mu')
        # z_logvar = tf.layers.dense(x, units=32, name='z_logvar')



        first_frame_pp = self.process_image(first_frame)

        eh.start_new_episode(first_frame_pp)

        while True:
            if np.random.rand() > self.get_epsilon():
                action_idx = self.session.run(
                    self.best_action,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]
            else:
                action_idx = self.get_random_action()

            if self.action_map is not None:
                action = self.action_map[action_idx]
            else:
                action = action_idx

            reward = 0
            for _ in range(self.frame_skip):
                observation, r, done, info = self.env.step(action)              
                if self.render:
                    self.env.render()
                reward += r
                if done:
                    break

            early_done, punishment = self.check_early_stop(reward, total_reward)
            if early_done:
                reward += punishment

            done = done or early_done

            total_reward += reward
            frames_in_episode += 1

            eh.add_experience(self.process_image(observation), action_idx, done, reward)

            if self.do_training:
                self.global_counter += 1
                if self.global_counter % self.network_update_freq:
                    self.update_target_network()
                train_cond = (
                    self.exp_history.counter >= self.min_experience_size and
                    self.global_counter % self.train_freq == 0
                )
                if train_cond:
                    self.train()

            if done:
                if self.do_training:
                    self.episode_counter += 1

                return total_reward, frames_in_episode

    def update_target_network(self):
        self.session.run(self.copy_network_ops)


class CarRacingDQN(DQN):
    """
    CarRacing specifig part of the DQN-agent
    Some minor env-specifig tweaks but overall
    assumes very little knowledge from the environment
    """

    def __init__(self, max_negative_rewards=100, **kwargs):
        all_actions = np.array(
            #[k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])]
            [k for k in it.product([-1, -0.5,-0.25, 0, 0.25, 0.5, 1], [1, 0], [0.2, 0])]
        )
        # car racing env gives wrong pictures without render
        kwargs["render"] = True
        super().__init__(
            action_map=all_actions,
            pic_size=(96, 96),
            **kwargs
        )

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] == 1 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards

    @staticmethod
    def process_image(obs):
        return 2 * color.rgb2gray(obs) - 1.0

    def get_random_action(self):
        """
        Here random actions prefer gas to break
        otherwise the car can never go anywhere.
        """
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)

        return np.random.choice(self.dim_actions, p=action_weights)

    def check_early_stop(self, reward, totalreward):
        if reward < 0:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0
