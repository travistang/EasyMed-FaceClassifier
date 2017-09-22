from FaceDataset import vgg16
from FaceDataset import utils
import numpy as np
import tensorflow as tf
import os
from random import shuffle
from itertools import permutations
batch_size = 4
#data_ratio = 0.5
vgg_out_dim = 4096
num_folds = 5
face_imgs_dir = 'FaceDataset/myDataBase'
nonface_imgs_dir = 'FaceDataset/google_things'

fc2_num_weights = 256
fin_out_dim = 2

# training config
num_epochs = 5

# This would be a generator that generates n pairs of generators.
# Each pair of generators will be training data generator and test data generator respectively.
def n_fold_batch_generator(n,face_img_path,nonface_img_path):
	assert n > 0
	get_imgs = lambda f: [ os.path.join(f,p) for p in os.listdir(f) if '.jpg' in p or '.png' in p]
	face_imgs = get_imgs(face_img_path)
	nonface_imgs = get_imgs(nonface_img_path)
	shuffle(face_imgs)
	shuffle(nonface_imgs)

	equal_slides = lambda l,s: [l[x * len(l) / s : (x + 1) * len(l) / s] for x in range(s)]
	# folds should be a list of pair of list of path
	# like [(face_img,non_face_img),(face_img,non_face_img)...]
	folds = zip(equal_slides(face_imgs,n),equal_slides(nonface_imgs,n))

	# fold should be a permutation of a list of pair of list of path...
	# There should be exactly n permutations (for n folds)
	# Each pair should be ([face_image paths],[non_face_image paths])
	# like [
	#	((face_img,non_face_img),(face_img,non_face_img)...),
	# 	(...),
	# ]
	for fold in permutations(folds):
		yield batch_generator(fold[:n - 1]), batch_generator([fold[n - 1]])

# create a generator given a list of folds
# list of folds should have the form [([a],[b])]
def batch_generator(folds):
	# now folds should have the form ([[a]],[[b]])
	folds = zip(*folds)
	concat = lambda a,b: a + b

	# map ... will have type ([a],[b])
	# so face_imgs and nonface_imgs with have type [a] and [b] respectively
	face_imgs,nonface_imgs = map(lambda l: reduce(concat,l),folds)
	num_face = len(face_imgs)
	num_nonface = len(nonface_imgs)
	print len(face_imgs),len(nonface_imgs)
	for i in range(min(len(face_imgs) / num_face,len(nonface_imgs) / num_nonface)):
		yield face_imgs[i * num_face: (i + 1) * num_face], nonface_imgs[i * num_nonface: (i + 1) * num_nonface]


model_input = tf.placeholder("float",[batch_size,224,224,3])
ans_input   = tf.placeholder('int32',[batch_size])
vgg = vgg16.Vgg16()
with tf.name_scope('content_vgg'):
	vgg.build(model_input)

# define the new fc layers here
with tf.name_scope('face_weights'):
	w1 = tf.get_variable("w1",[vgg_out_dim,fc2_num_weights],initializer = tf.random_normal_initializer(stddev = 1e-4))
	b1 = tf.get_variable("b1",initializer = tf.zeros([fc2_num_weights]))

	w2 = tf.get_variable("w2",[fc2_num_weights,fin_out_dim],initializer = tf.random_normal_initializer(stddev = 1e-4))
	b2 = tf.get_variable("b2",initializer = tf.zeros([fin_out_dim]))

	# res of the computation graphs
	vgg_oup = vgg.fc6
	# first fc
	fc2 = tf.nn.relu(tf.nn.dropout(tf.nn.xw_plus_b(vgg_oup,w1,b1),0.8))
	# second (and last) fc
	fin_out = tf.nn.relu(tf.nn.xw_plus_b(fc2,w2,b2))
	# train op
	loss 	= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = ans_input,logits = fin_out,name = 'loss'))
	accuracy= tf.reduce_mean(tf.abs(tf.argmax(tf.nn.softmax(fin_out),axis = -1) - ans_input),name = 'accuracy')
	opt 	= tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss,var_list = [w1,w2,b1,b2])

	tf.summary.scalar('loss',loss)
	tf.summary.scalar('accuracy',accuracy)
	tf.summary.histogram('output',fin_out)

	summary = tf.summary.merge_all()
	saver = tf.train.Saver()

def main():
	with tf.Session() as sess:
                try:
                    writer = tf.summary.FileWriter('/FaceDataset/tmp/face',sess.graph)
                    sess.run(tf.global_variables_initializer())
                    __i = 0
					'''
						Each epoch performs one cycle of 5-fold Training
						Test will be performed when a fold ends
					'''
					def make_training_tensors(face_imgs_path,non_face_imgs_path):
							batch = map(utils.load_image,face_imgs_path + non_face_imgs_path):
							lbls = np.zeros(len(batch))
							lbls[:len(face_batch)] = 1.
							batch = np.stack(batch,axis = 0)
                            assert batch.shape == (batch_size,224,224,3)
                            assert lbls.shape == (batch_size,)
							return batch,lbls

                    for epoch in range(num_epochs):
						for train_gen,test_gen in n_fold_batch_generator(num_folds,face_imgs_dir,nonface_imgs,dir):
							# perform training on the training set
							for face_imgs_path,nonface_imgs_path in train_gen:
								input_batch,lbls = make_training_tensors(face_imgs_path,nonface_imgs_path)
								summary_val,out_val,loss_val,acc_val,_ = sess.run([
									summary,fin_out,loss,accuracy,opt],
									feed_dict = {
										ans_input: lbls,
										model_input: input_batch
									})
								# logging and stats and stuff
								writer.add_summary(summary_val,__i)
								__i += 1
                                print 'epoch: {}; loss: {}'.format(epoch,loss_val)
							# perform validation on testing set
							for face_imgs_path,nonface_imgs_path in test_gen:
								input_batch,lbls = make_training_tensors(face_imgs_path,nonface_imgs_path)
								summary_val,out_val,loss_val,acc_val = sess.run([
									summary,fin_out,loss_val,acc_val],
									feed_dict = {
										ans_input: lbls,
										model_input: input_batch
									})
								writer.add_summary(summary_val,__i)
								__i += 1
								print 'epoch: {}; test loss: {}'.format(epoch,loss_val)

#                            for face_batch,non_face_batch in batch_generator('FaceDataset/myDataBase','FaceDataset/google_things'):
#
#                                    #print 'here'
#                                    summary_val, out_val,loss_val,_ = sess.run([summary,fin_out,loss,opt],feed_dict = {ans_input: lbls,model_input: batch})
#                                    writer.add_summary(summary_val,__i)
#                                    __i += 1
#                                    print 'epoch: {}; loss: {}'.format(epoch,loss_val)
#                                    # print '\r'
							w1_val = w1.eval()
							w2_val = w2.eval()
							b1_val = b1.eval()
							b2_val = b2.eval()

                            np.save('w1',w1_val)
                            np.save('w2',w2_val)
                            np.save('b1',b1_val)
                            np.save('b2',b2_val)

                            print '\n'

                except KeyboardInterrupt:
                    pass
#                saver.save(sess,'tmp/face_classifier.ckpt')
                print 'Training complete. Model saved'

if __name__ == '__main__':
	face_img_path = 'FaceDataset/myDataBase'
	non_face_batch = 'FaceDataset/google_things'
	for train_gen,test_gen in n_fold_batch_generator(5,face_img_path,non_face_batch):
		train_gen.next()
	#main()
