#!/usr/bin/python
import re
import argparse
import tensorflow as tf
'''
	Some functionalities:
		1. Check the list of tensors of the checkpoint
		2. Check the list of operations of the given checkpoint

		Usage:
			checkpoint.py -c <checkpoint_dir.ckpt> -t [<search string>] -o [<search string>]
			-c: checkpoint dir
			-t: list tensors
			-o: list operations
'''

# functions:
class Checkpoint:
	def __init__(self,checkpoint_dir):
		if not re.search(r'\.ckpt$',checkpoint_dir):
			print 'Checkpoint path must end with .ckpt'
			exit(1)
		self.dir = checkpoint_dir
		self.meta_graph = self.dir + '.meta'
		print 'Loading meta graph...'
		self.sess = tf.Session()
		self.saver = tf.train.import_meta_graph(self.meta_graph)
		print 'Restoring session...'
		self.saver.restore(self.sess,self.dir)

	def search_tensor(self,name = None):
		def pprintTensor(tensors):
			for t in tensors:
				print "{} : {}".format(t.name,t.shape)
		if not name:
			print 'List of tensors:'
			pprintTensor(tf.global_variables())
		else:
			print 'List of tensor with substring: ' + name
			pprintTensor([t for t in tf.global_variables if name in t.name])

	def search_operations(self,name = None):
		def pprintOp(op):
			for o in op:
				print "{} : {}".format(o.name,o.outputs)
		if not name:
			print "List of operations:"
			pprintOp(self.sess.graph.get_operations())
		else:
			print 'List of operations with subtring: ' + name
			pprintOp([o for o in self.sess.graph.get_operations() if name in o.name])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--checkpoint',help = 'path to checkpoint of your model',required = True)
	parser.add_argument('-t','--tensor',help = 'tensor name')
	parser.add_argument('-o','--operation',help = 'operation name')
	args = parser.parse_args()
	cp = Checkpoint(args.checkpoint)
	if args.tensor == "list-all":
		cp.search_tensor()
        elif args.tensor is not None:
		cp.search_tensor(args.tensor)

	if args.operation == "list-all":
		cp.search_operations()
        elif args.operation is not None:
		cp.search_operations(args.operation)
