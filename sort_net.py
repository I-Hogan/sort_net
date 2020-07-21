# implementation of "A NEURAL SORTING NETWORK WITH O(1) TIME COMPLEXITY"

import os
import math
import random
import time

import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
print("\nTensorFlow Version:", tf.__version__, "\n")



#---------- Functions ----------

# returns the xth permutation of list N where x ɛ [1, ... ,n!]
# complexity: O(n)
def permutation(x_,N_):
	x  = x_							#avoid alias
	N = list(N_)					#avoid alias
	n = len(N)
	permutation = []
	for i in range(n):
		#ith_rank = x % n
		permutation.append(N[x % n])
		del N[x % n]
		x = x // n
		n = n - 1
	return permutation
	
	
#---------- sampling functions ----------

# O(n)
def sort_sampling(n, min, max, batch_size):
	X, Y = [], []
	for i in range(batch_size):
		X.append([float(random.randint(min, max)) for i in range(n)])
	for x in X:
		Y.append(sorted(x))
	return tf.constant(X), tf.constant(Y)
	
#---------- Custom Block ----------

class Dense_Attention_Block(tf.keras.Model):
	def __init__(self, num_outputs):
		super(Dense_Attention_Block, self).__init__()
		self.num_outputs = num_outputs
		self.d_query = tf.keras.layers.Dense(self.num_outputs, input_dim=(-1,self.num_outputs), activation=tf.nn.relu)
		self.d_key = tf.keras.layers.Dense(self.num_outputs, activation=tf.nn.relu)
		self.d_val = tf.keras.layers.Dense(self.num_outputs, activation=tf.nn.relu)
		self.d1 = tf.keras.layers.Dense(self.num_outputs)
	def call(self, input):
		query = self.d_query(input)
		key = self.d_key(input)
		value = self.d_val(input)
		attention = tf.nn.softmax(tf.math.multiply(query,key), axis=1)
		output = self.d1(tf.math.multiply(attention, value))
		return output


#---------- Models ----------

class Hard_Coded_Selection_Net(tf.keras.Model):
	def __init__(self, n):
		super(Hard_Coded_Selection_Net, self).__init__()
		
		#rank
		self.Q1 = tf.ones([n,])
		self.Qb1 = tf.zeros([n,])
		self.R1 = tf.zeros([n,])
		self.Rb1 = tf.range(0.,n,1)
		s1 = np.zeros((n,))
		s1.fill(-1.)
		s1[0] = -1.
		self.S1 = tf.convert_to_tensor(s1)
		self.e1 = 1.
		
		#order
		self.Q3 = tf.ones([n,])
		self.Qb3 = tf.zeros([n,])
		self.R3 = tf.zeros([n,])
		self.Rb3 = tf.zeros([n,])
		s3 = np.zeros((n,))
		s3.fill(-1.)
		s3[0] = -1.
		self.S3 = tf.convert_to_tensor(s1)
		self.e3 = 0.
		self.Eb3 = tf.range(0.,n,1)
		
		
		#select tf.range(0.,n,1)
		q2 = np.zeros((2,n))
		q2[1] = 1.
		self.Q2 = tf.cast(tf.convert_to_tensor(q2), tf.float32)
		self.Qb2 = tf.zeros([n,])
		r2 = np.zeros((2,n))
		r2[0] = 1.
		self.R2 = tf.cast(tf.convert_to_tensor(r2), tf.float32)
		self.Rb2 = tf.zeros([n,])
		s2 = np.zeros((n,))
		s2.fill(0.)
		s2[0] = 1.
		self.S2 = tf.convert_to_tensor(s2)
		self.e2 = 0.
		
		#tf.print("self.Q1:", self.Q1)
		#tf.print("self.Qb1:", self.Qb1)
		#tf.print("self.R1:", self.R1)
		#tf.print("self.Rb1:", self.Rb1)
		#tf.print("self.S1:", self.S1)
		
		#tf.print("self.Q2:", self.Q2)
		#tf.print("self.Qb2:", self.Qb2)
		#tf.print("self.R2:", self.R2)
		#tf.print("self.Rb2:", self.Rb2)
		#tf.print("self.S2:", self.S2)
		
	def call(self, X):
		
		X = X[0]
		
		#rank
		Z1 = tf.math.add(self.Q1 * X, self.Qb1)
		P1 = tf.math.add(self.R1 * X, self.Rb1)
		Y_1 = []
		for i in range(n):
			SP1 = tf.roll(self.S1, shift=tf.cast(P1[i], tf.int32), axis=0)
			Y_int1 = tf.math.add(tf.cast(SP1, tf.float32) * Z1, Z1[i] * self.e1)
			Y_act1 = tf.math.sign(tf.nn.relu(Y_int1))
			Y_1.append(tf.math.reduce_sum(Y_act1))
		Y_1 = tf.convert_to_tensor(Y_1)
		
		#tf.print("\nrank:", Y_1)
		
		#order
		Z3 = tf.math.add(self.Q3 * Y_1, self.Qb3)
		P3 = tf.math.add(self.R3 * Y_1, self.Rb3)
		Y_3 = []
		for i in range(n):
			SP3 = tf.roll(self.S3, shift=tf.cast(P3[i], tf.int32), axis=0)
			Y_int3 = tf.math.add(tf.cast(SP3, tf.float32) * Z3, Z3[i] * self.e3 + self.Eb3[i])
			Y_act3 = (tf.math.abs(tf.math.sign(Y_int3)) * (-1) + 1) * self.Eb3
			Y_3.append(tf.math.reduce_sum(Y_act3))
		Y_3 = tf.convert_to_tensor(Y_3)
		
		#tf.print("\norder:", Y_3)
		
		#select
		X_2 = tf.concat([[Y_3], [X]], 0)
		Z2 = tf.math.add(tf.math.reduce_sum(self.Q2 * X_2, axis=0), self.Qb2)
		P2 = tf.math.add(tf.math.reduce_sum(self.R2 * X_2, axis=0), self.Rb2)
		Y_2 = []
		for i in range(n):
			SP2 = tf.roll(self.S2, shift=tf.cast(P2[i], tf.int32), axis=0)
			Y_int2 = tf.math.add(tf.cast(SP2, tf.float32) * Z2, Z2[i] * self.e2)
			Y_act2 = Y_int2
			Y_2.append(tf.math.reduce_sum(Y_act2))
		Y_2 = tf.convert_to_tensor(Y_2)
		
		#tf.print("\nsorted:", Y_2)
		
		return Y_2
		
"""
	97	5	42	82	59
0	0	1	1	1	1
1	0	0	0	0	0
2	0	1	0	0	0
3	0	1	1	0	1
4	0	1	1	0	0
"""	

class Dense_Net(tf.keras.Model): 
	def __init__(self):
		super(Dense_Net, self).__init__()
		self.d1 = tf.keras.layers.Dense(n, input_dim=(-1,n), activation=tf.nn.relu)
		self.d2 = tf.keras.layers.Dense(n, activation=tf.nn.relu)
		self.d3 = tf.keras.layers.Dense(n**2, activation=tf.nn.relu)
		self.d4 = tf.keras.layers.Dense(n, activation=None)
	def call(self, x):
		rank_sum = self.d1(x)
		rank = self.d2(rank_sum)
		rank_comb = self.d3(tf.concat([rank, x], axis=1))
		extraction = self.d4(rank_comb)
		return extraction
		
class Attention_Net(tf.keras.Model): 
	def __init__(self):
		super(Attention_Net, self).__init__()
		
		#attention block 1
		self.d_query1 = tf.keras.layers.Dense(n, input_dim=(-1,n), activation=tf.nn.relu)
		self.d_key1 = tf.keras.layers.Dense(n, activation=tf.nn.relu)
		self.d_val1 = tf.keras.layers.Dense(n, activation=tf.nn.relu)
		self.d_1 = tf.keras.layers.Dense(n)
		
		#attention block 2
		self.d_query2 = tf.keras.layers.Dense(n, activation=tf.nn.relu)
		self.d_key2 = tf.keras.layers.Dense(n, activation=tf.nn.relu)
		self.d_val2 = tf.keras.layers.Dense(n, activation=tf.nn.relu)
		self.d_2 = tf.keras.layers.Dense(n)

	def call(self, input):
		
		#attention block 1
		query1 = self.d_query1(input)
		key1 = self.d_key1(input)
		value1 = self.d_val1(input)
		attention1 = tf.nn.softmax(tf.math.multiply(query1,key1), axis=1)
		output1 = self.d_1(tf.math.multiply(attention1, value1))
		
		#attention block 2
		#input2 = tf.concat([output1, input], axis=1)
		input2 = output1
		query2 = self.d_query2(input2)
		key2 = self.d_key2(input2)
		value2 = self.d_val2(input2)
		attention2 = tf.nn.softmax(tf.math.multiply(query2,key2), axis=1)
		output2 = self.d_2(tf.math.multiply(attention2, value2))		
		
		#extraction = self.a1(tf.concat([rank, x], axis=1))
		#extraction = self.a1(rank)
		return output2
		
#---------- Training Step ----------

def train_step(n, sample_function):
	with tf.GradientTape() as tape:
		X, Y = sample_function(n, min_val, max_val, batch_size)
		Y_ = model(X)
		loss = loss_obj(Y, Y_)
	gradients = tape.gradient(loss,model.trainable_variables)
	optimizer.apply_gradients(zip(gradients,model.trainable_variables))		
	return loss
	

#---------- Tests ----------

# set parameters
n = 10000			# length of the sequence to be sorted

min_val = 0			# minimum value for the list
max_val = 1000000		# maximum value for the list

epochs = 100
epoch_size = 10000000
batch_size = 1000
l_rate = 10**-2

model_type = 2

#n_fact = math.factorial(n)
#for i in range(n_fact):
#	print(permutation(i,range(n)))
	
"""
	
loss_obj = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.SGD(learning_rate=l_rate)

if model_type == 1:
	model_type = Dense_Net
	model_name = "dense_sort_1.h5"
elif model_type == 2:
	model_type = Attention_Net
	model_name = "attention_sort_1.h5"

try:
	print()
	model = model_type()
	model.build(tuple([batch_size,n]))
	if force_new_model:
		print("creating model:", model_name)
	else:
		model.load_weights(model_name)
		print("\nloading model:", model_name)
	model.summary()
except:
	print("creating model:", model_name)
	model = model_type()
	model.build(tuple([batch_size,n]))
	model.summary()

sample_function = sort_sampling


print("training model...")

for i in range(epochs):
	loss = 0
	for j in range(epoch_size//batch_size):
		loss += train_step(n, sample_function)
	model.save_weights(model_name)
	tf.print("epoch:", i + 1, "\tloss:", loss)
	X, Y = sample_function(n, min_val, max_val, 1)
	Y_ = model(X)
	tf.print("X:", X, "\tY:", Y, "\tY_:", Y_)
"""

model = Hard_Coded_Selection_Net(n)
sample_function = sort_sampling
	
print("testing model...")

python_time = 0
tensor_time = 0

for i in range(10):
	start_time = time.time()
	X, Y = sample_function(n, min_val, max_val, 1)
	python_time +=  time.time() - start_time
	start_time = time.time()
	Y_ = model(X)
	tensor_time +=  time.time() - start_time
	tf.print("X:", X, "\tY:", Y, "\tY_:", Y_)
	
print("python_time: ", python_time)
print("tensor_time: ", tensor_time)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

