from FaceDataset import vgg16
from FaceDataset import utils
import numpy as np
import tensorflow as tf
import os
from random import shuffle

batch_size = 4
data_ratio = 0.5
vgg_out_dim = 4096

fc2_num_weights = 256
fin_out_dim = 2

# training config
num_epochs = 100


def batch_generator(face_img_path,nonface_img_path):
	get_imgs = lambda f: [ os.path.join(f,p) for p in os.listdir(f) if '.jpg' in p or '.png' in p]
	face_imgs = get_imgs(face_img_path)
	nonface_imgs = get_imgs(nonface_img_path)
	# then shuffle the image paths
	shuffle(face_imgs)
	shuffle(nonface_imgs)

	num_face = int(batch_size * data_ratio)
	num_nonface = batch_size - num_face
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

	opt = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss,var_list = [w1,w2,b1,b2])
def main():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(num_epochs):

			for face_batch,non_face_batch in batch_generator('FaceDataset/faces94','FaceDataset/google_things'):
				batch = map(utils.load_image,face_batch + non_face_batch)
				#print 'here'
				lbls = np.zeros(len(batch))
				# make labels...
				lbls[:len(face_batch)] = 1.
				batch = np.stack(batch,axis = 0)
					
				#print 'here'
				assert batch.shape == (batch_size,224,224,3)
				assert lbls.shape == (batch_size,)
				
				#print 'here'
				out_val,loss_val,_ = sess.run([fin_out,loss,opt],feed_dict = {ans_input: lbls,model_input: batch})
				
				#print 'here'
				print 'epoch: {}; loss: {}'.format(epoch,loss_val),
				print '\r',
				#print 'output: {}'.format(out_val)	
			print '\n'

if __name__ == '__main__':
	main()
