import numpy as np
from PIL import Image
import os

def sigmoid(z): # z is vector
	return 1.0/(1.0+np.exp(-z)) # np.exp() automatically expands to vector

def sigmoid_prime(z): # z is vector
	return sigmoid(z)*(1-sigmoid(z))


def open_image(name):
	img = Image.open(name, "r")
	gs_img = img.convert("L") # L means gray scale ?
	img_array = np.asarray(gs_img)
	print "Input data type is %s" % img_array.dtype
	answer = int(os.path.basename(name)[0])
	return (np.reshape(img_array, (32*32,1)))/255.0, answer # <- Dividing max is more smart

def learn_once(input_array, answer, objective):
	num_layers = len(sizes)

	biases = [np.random.randn(x,1) for x in sizes[1:]] # Eliminate biases of the first(input) layer
	weights = [np.random.randn(y,x) for(x,y) in zip(sizes[:-1], sizes[1:])] 

	#### Output info. of weights
	for loop_num, w in enumerate(weights):
		print("%d: %s" %(loop_num, w.shape))

	eta = 0.1
	loop_limit = 2
	while(1):
	#for loop in xrange(loop_limit): # Loop until loss becomes lower than objective or upto loop_limit
		####  Calclate output
		a = input_array # Initialize
		for i in range(len(weights)): a = sigmoid(np.dot(weights[i], a)+biases[i])

		answer_array = np.zeros((10, 1), dtype=np.float64)
		answer_array[answer][0] = 1.0

		loss = 0.0
		loss_array = np.square(a-answer_array)
		for l in loss_array.flatten(): loss += l
		if(loss < objective): break
		else:
			for i in range(len(weights)):  = a-eta*sigmoid_prime(np.dot(weights[i], a)+biases[i])
			 

	return loss, weights, biases	


sizes = [32*32, 15, 10]
objective = 0.1
for i in range(10):
	input_array, answer = open_image("./"+str(i)+"_aaa.png")
	loss, trained_ws, trained_bs = learn_once(input_array, answer, objective)
	print loss

