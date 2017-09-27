#import skimage
#import skimage.io
#import skimage.transform
import cv2
import numpy as np
from data_augmentation import *
# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
	try:
		if '.pgm' in path:
			img = read_pgm_to_array(path)
		else:
			img = cv2.imread(path)
			img = data_augmentation([img],{
            	'horizontal_flips': True,
                'gaussian_blur': True,
    		})[0]
		img = img / 255.0
		return cv2.resize(img,(224,224))
	except TypeError:
		print path
		raise
    #return resized_img

# ref:https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
def read_pgm_to_array(path,byteorder = '>'):
	with open(path, 'rb') as f:
		buffer = f.read()
		try:
			header, width, height, maxval = re.search(
			    b"(^P5\s(?:\s*#.*[\r\n])*"
			    b"(\d+)\s(?:\s*#.*[\r\n])*"
			    b"(\d+)\s(?:\s*#.*[\r\n])*"
			    b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
		except AttributeError:
			raise ValueError("Not a raw PGM file: '%s'" % filename)
			img = numpy.frombuffer(buffer,
			                    dtype='u1' if int(maxval) < 256 else byteorder+'u2',
			                    count=int(width)*int(height),
			                    offset=len(header)
			                    ).reshape((int(height), int(width)))
	# formatting image...
	return img

## returns the top1 string
#def print_prob(prob, file_path):
#    synset = [l.strip() for l in open(file_path).readlines()]
#
#    # print prob
#    pred = np.argsort(prob)[::-1]
#
#    # Get top1 label
#    top1 = synset[pred[0]]
#    print(("Top1: ", top1, prob[pred[0]]))
#    # Get top5 label
#    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
#    print(("Top5: ", top5))
#    return top1
#
#
#def load_image2(path, height=None, width=None):
#    # load image
#    img = skimage.io.imread(path)
#    img = img / 255.0
#    if height is not None and width is not None:
#        ny = height
#        nx = width
#    elif height is not None:
#        ny = height
#        nx = img.shape[1] * ny / img.shape[0]
#    elif width is not None:
#        nx = width
#        ny = img.shape[0] * nx / img.shape[1]
#    else:
#        ny = img.shape[0]
#        nx = img.shape[1]
#    return skimage.transform.resize(img, (ny, nx))
#

def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)

def test_net(path,checkpoint_path,input_op_name,output_op_name):
	import tensorflow as tf
	img = load_image(path)
	img = np.expand_dims(img,0) # expand the first dimension of the tensor to be the batch_size of it
	with tf.Session() as sess:
		loader = tf.train.import_checkpoint_path(checkpoint_path + '.meta')

		loader.restore(sess,checkpoint_path)
		input_op = sess.graph.get_tensor_by_name(input_op_name)
		output_op = sess.graph.get_operation_by_name(output_op_name)
		return sess.run(output_op,feed_dict = {input_op: img})

def test_face_classifier(face_path,nonface_path,checkpoint_dir):
	import tensorflow as tf
	# try each face and non_face images
	draw_pic = lambda path: choice([i for i in path if '.jpg' in i or '.png' in i])
	face_img = draw_pic(face_path)
	nonface_img = draw_pic(nonface_path)

	res_dict = {0: 'face',1: 'non-face'}
	print 'testing face image...'

	print 'The model detected face image as: {}'.format(
		res_dict[test_net(
			face_img,
			checkpoint_dir,
			'model_input:0',
			'face_weights/model_output')])
	print 'testing non-face image'
	print 'The model detected non-face image as: {}'.format(
		res_dict[test_net(
			nonface_img,
			checkpoint_dir,
			'model_input:0',
			'face_weights/model_output')])


def parse_eye_file(path):
	org_x,org_y = 384,286

	with open(path) as f:
		pos = f.readlines()[-1].strip().split()
		# return the normalized coordinates of the eyes
		return np.array(map(lambda (p,dim): float(p) / dim,zip(map(int,pos),[org_x,org_y] * 2)))

if __name__ == "__main__":
	test_face_classifier('FaceDataset/face94','FaceDataset/google_things')
