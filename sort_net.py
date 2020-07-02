# implementation of "A NEURAL SORTING NETWORK WITH O(1) TIME COMPLEXITY"

import os
import math
import random


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
n = 5				# length of the sequence to be sorted

min_val = 0			# minimum value for the list
max_val = 100		# maximum value for the list

epochs = 100
epoch_size = 10000000
batch_size = 1000
l_rate = 10**-2

model_type = 2

#n_fact = math.factorial(n)
#for i in range(n_fact):
#	print(permutation(i,range(n)))
	
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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

