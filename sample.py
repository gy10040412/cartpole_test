import numpy as np
import random
import tensorflow as tf
import environment
from collections import deque

import gym

env = gym.make('CartPole-v0')
env._max_episode_steps = 10000

input_size= env.observation_space.shape[0]
output_size=env.action_space.n

dis = 0.9
REPLAY_MEMORY = 5000

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session=session
        self.input_size=input_size
        self.output_size=output_size
        self.net_name=name
        self._build_network()
    def _build_network(self,l_rate=1e-3):
        h_size=(self.input_size>self.output_size)*self.input_size+(self.input_size<self.output_size)*self.output_size
        print (h_size)
        with tf.variable_scope(self.net_name):
            self._X=tf.placeholder(
                tf.float32,[None,self.input_size],name="input_x")
            
            # First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size,h_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X,W1))
            
            # Second layer of weights
            W2 = tf.get_variable("W2", shape=[h_size,self.output_size],
                                initializer=tf.contrib.layers.xavier_initializer())

            # Q prediction
            self._Qpred = tf.matmul(layer1,W2)

        # Define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32)
    
        # Cost Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)
    
    def predict(self, state):
        x =np.reshape(state, [1,self.input_size])
        return self.session.run(self._Qpred,feed_dict={self._X: x})
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

#------------------------------------------------------------------
def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    #get information from buffer
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)

        # terminated?
        if done :
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis *np.max(DQN.predict(next_state))
        
        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])
    return DQN.update(x_stack, y_stack)

#------------------------------------------------------------------
def bot_play(mainDQN):
    # play with trained network
    s= env.reset()
    reward_sum=0

    while True:
        # env.render()
        a= np.argmax(mainDQN.predict(s))
        s,rewrad,done,info=env.step(a)
        reward_sum += reward
        if done:
            print (env.logg,reward_sum)
            break
    print ("reward: {}".format(reward_sum))
    
#------------------------------------------------------------------

def get_copy_var_ops(*, dest_scope_name="target",src_scope_name="main"):
    #Copy variables src_scop to dest_scrop
    op_holder=[]

    src_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    
    #달라진 w만 copy해서 변경해주는것

    return op_holder
#------------------------------------------------------------------
def replay_train(mainDQN, targetDQN,train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    #Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q= mainDQN.predict(state)

        #terminated
        if done:
            Q[0, action] = reward
        else:
            #get target from target DQN(Q')
            Q[0, action] = reward + dis *np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])
    
    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)




#------------------------------------------------------------------

if __name__ == '__main__':

    max_episodes = 5000
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess,input_size,output_size,name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        sess.run(copy_ops)

        roop_step_count=0
        for episode in range(max_episodes):
            e = 1./ ((episode/10)+1)
            done = False
            step_count =0
            state = env.reset()

            while not done:
                if np.random.rand(1)<e:
                    action = env.action_space.sample()
                else:
                    #Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))
                
                #Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                
                if done: #penalty
                    reward = -100
                
                #save the experience to our buffer
                replay_buffer.append((state,action,reward,next_state,done))
                if len(replay_buffer)>REPLAY_MEMORY:
                    replay_buffer.popleft()
                
                #################################################
                try:
                    minibatch = random.sample(replay_buffer, 32)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                except:
                    pass
                roop_step_count+=1
                if (roop_step_count%50) ==0:
                    sess.run(copy_ops)
                #################################################
                state = next_state
                step_count+=1
                
                if step_count > 10000:
                    break
            print ("Episode : {} steps : {}".format(episode, step_count))
            if step_count>10000:
                pass
        

        bot_play(mainDQN)
