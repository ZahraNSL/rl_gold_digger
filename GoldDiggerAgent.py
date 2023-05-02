import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model

'''
The A2C class defines the Actor (predicts the probabilities of each action given the state) and 
Critic models (predicts the value function of the state). 
Also provides methods to train the models (train()) and act in the environment (act()). 

'''
class A2C:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.actor.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
        self.critic.compile(optimizer='adam', loss="mse", metrics=["accuracy"])

    def _build_actor(self):
        # Build actor model
        state_input = Input(shape=(self.state_size,))
        x = Dense(32, activation='relu')(state_input)
        #x = Dense(5, activation='relu')(x)
        action_probs = Dense(self.action_size, activation='softmax')(x)
        actor = Model(inputs=state_input, outputs=action_probs)

        return actor

    def _build_critic(self):
        # Build critic model
        state_input = Input(shape=(self.state_size,))
        x = Dense(32, activation='relu')(state_input)
        #x = Dense(5, activation='relu')(x)
        state_value = Dense(1, activation='tanh')(x)
        critic = Model(inputs=state_input, outputs=state_value)

        return critic

    def act(self, state):

        probs = self.actor.predict(np.reshape(state, (1, self.state_size)))[0]
        action = np.random.choice(self.action_size, p=probs)
        return action

    def train(self, states, actions, rewards, dones, next_states):
        # Compute returns
        returns = self._compute_returns(rewards, dones, next_states)

        # Compute advantages
        # Using them to update the actor and critic networks
        values = []
        for state in states:
            value = self.critic.predict(np.array([state]))[0, 0]
            values.append(value)

        advantages = returns - np.array(values)

        # Train actor and critic
        self._train_actor(states, actions, advantages)
        self._train_critic(states, returns)

    def _compute_returns(self, rewards, dones, next_states):
        returns = np.zeros_like(rewards)
        value = self.critic.predict(np.array([next_states[-1]]))[0, 0] if not dones[-1] else 0.0

        for t in reversed(range(len(rewards))):
            value = rewards[t] + self.gamma * value * (1 - dones[t])
            returns[t] = value

        return returns.tolist()

    def _train_actor(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            # Compute logits and loss
            probs = self.actor(np.reshape(states, (len(states), self.state_size)))
            log_probs = tf.math.log(tf.reduce_sum(probs * tf.one_hot(actions, self.action_size), axis=1, keepdims=True))
            actor_loss = -tf.reduce_mean(log_probs * advantages)

        # Compute gradients and update the weights of the actor
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def _train_critic(self, states, returns):
        with tf.GradientTape() as tape:
            # Compute MSE loss
            values = self.critic(np.reshape(states, (len(states), self.state_size)))
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        # Compute gradients and update the weights of the critic
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
