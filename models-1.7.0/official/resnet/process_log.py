import sys
import argparse
import tensorflow as tf
from cifar10_main import input_fn 
from cifar10_main import Cifar10Model

'''
usage: 
for sequential training:
	python process_log.py -m='svd' -r=0.1 -fn='/home/jingling/Logs/seq_non_reshaped/processed/svd_0.1.log' -tm='seq' 

for end to end training:
	python process_log.py -m='svd' -r=0.1 -fn='/home/jingling/Logs/e2e_non_reshaped/processed/svd_0.1.log' -tm='e2e'

'''

def list_multiply(arr):
	res = 1
	for a in arr:
		res = res*a
	return res

def calculate_total_params(vars_list):
	total = 0
	for v in vars_list:
		cur_params = list_multiply(v.shape.as_list())
		total += cur_params
	return total

def main(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--fname', '-fn')
	parser.add_argument('--method', '-m')
	parser.add_argument('--rate', '-r', type=float)
	parser.add_argument('--batch_size', '-bs', type=int)
	parser.add_argument('--train_method', '-tm')
	flags = parser.parse_args(args=argv[1:])
	data_dir='/home/jingling/Data/cifar10_data/'
	
	fname, method, rate, train_method = flags.fname, flags.method, flags.rate, flags.train_method

	dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=128)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	model = Cifar10Model(resnet_size=32, method=method, scope=method, rate=rate)
	model(next_element[0], False)

	var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	
	conv_sublists = []

	for i in range(3):
		conv_sublist = []
		if i==0:
			conv_sublist = [var_list[0]]
		conv_sublist.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "%s/block%d" %(method, i)))
		conv_sublists.append(conv_sublist)

	with open(fname) as f:
		content = f.readlines()

	accuracies = []
	num_gradient_updates = [] # in terms of million
	last_epoch = 0
	if train_method == 'seq':
		total_params = []
		for arr in conv_sublists:
			total_params.append(calculate_total_params(arr))

		cur_block = 0
		gradient_updates = 0
		for c in content:
			if 'block' in c:
				cur_block += 1
				last_epoch = 0
				continue	
			line = c.split(' ')
			
			epoch = int(line[1].strip(','))
			epoch_gap = epoch - last_epoch

			test_acc = float(line[-1])
			gradient_updates += int(epoch_gap*5*total_params[cur_block]/100)
			num_gradient_updates.append(gradient_updates)
			accuracies.append(test_acc)

			last_epoch = epoch

	elif train_method == 'e2e':
		total_params = calculate_total_params(var_list)
		gradient_updates = 0

		for c in content:
			line = c.split(' ')
			epoch = int(line[4].strip(':'))
			epoch_gap = epoch - last_epoch

			test_acc = float(line[6].strip(','))

			gradient_updates += int(epoch_gap*5*total_params/100)
			num_gradient_updates.append(gradient_updates)
			accuracies.append(test_acc)

			last_epoch = epoch

	print('grad_ups =', num_gradient_updates, ';') 
	print('accs =', accuracies, ';')


if __name__ == '__main__':
	main(argv=sys.argv)