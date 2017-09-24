import numpy as np
import cv2


def data_augmentation(images, options={}):
	horizontal_flips = options.get('horizontal_flips', False)
	distortions = options.get('distortions', False)
	stretching = options.get('stretching', False)
	random_scales = options.get('random_scales', False)
	color_jitter = options.get('color_jitter', False)
	shear = options.get('shear', False)
	inverse = options.get('inverse', False)
	sobel_derivative = options.get('sobel_derivative', False)
	scharr_derivative = options.get('scharr_derivative', False)
	laplacian = options.get('laplacian', False)
	blur = options.get('blur', False)

        uni_prob = lambda p: np.random.random() < p
	blur_config = options.get('blur_config', {
		'kernel_size': 15,
		'step_size': 2
	})
	gaussian_blur = options.get('gaussian_blur', False)
	gaussian_blur_config = options.get('gaussian_blur_config', {
		'kernel_size': 20,
		'step_size': 2
	})
	median_blur = options.get('median_blur', False)
	median_blur_config = options.get('median_blur_config', {
		'kernel_size': 10,
		'step_size': 2
	})
	bilateral_blur = options.get('bilateral_blur', False)
	bilateral_blur_config = options.get('bilateral_blur_config', {
		'kernel_size': 30,
		'step_size': 2
	})
	shuffle_result = options.get('shuffle_result', False)

	augmented_images_set = images[:]

	# TODO
	if uni_prob(0.5) and  distortions:
		augmented_images_set += []

	# TODO
	if uni_prob(0.5) and  stretching:
		augmented_images_set += []

	# TODO
	if uni_prob(0.5) and  random_scales:
		augmented_images_set += []

	# TODO
	if uni_prob(0.5) and  color_jitter:
		augmented_images_set += []

	# TODO
	if uni_prob(0.5) and  shear:
		augmented_images_set += []

	if inverse:
		augmented_images_set += [(255 - image) for image in images]

	if uni_prob(0.5) and  sobel_derivative:
		derivatives = []
		for image in images:
			image = cv2.GaussianBlur(image, (3, 3), 0)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			grad_x = cv2.Sobel(
				gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
			)
			grad_y = cv2.Sobel(
				gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
			)
			abs_grad_x = cv2.convertScaleAbs(grad_x)
			abs_grad_y = cv2.convertScaleAbs(grad_y)
			dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
			derivatives.append(dst)
		augmented_images_set += derivatives

	if uni_prob(0.5) and  scharr_derivative:
		derivatives = []
		for image in images:
			image = cv2.GaussianBlur(image, (3, 3), 0)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			grad_x = cv2.Scharr(gray, cv2.CV_16S, 1, 0)
			grad_y = cv2.Scharr(gray, cv2.CV_16S, 0, 1)
			abs_grad_x = cv2.convertScaleAbs(grad_x)
			abs_grad_y = cv2.convertScaleAbs(grad_y)
			dst = cv2.add(abs_grad_x, abs_grad_y)
			derivatives.append(dst)
		augmented_images_set += derivatives

	if uni_prob(0.5) and  laplacian:
		laplacians = []
		for image in images:
			image = cv2.GaussianBlur(image, (3, 3), 0)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray_lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3, scale=1, delta=0)
			dst = cv2.convertScaleAbs(gray_lap)
			laplacians.append(dst)
		augmented_images_set += laplacians

	if uni_prob(0.5) and  blur:
		augmented_images_set += np.hstack([
				[
					cv2.blur(image, (i, i))
					for i in xrange(1, blur_config['kernel_size'], blur_config['step_size'])
				]
				for image in images
			]).tolist()

	if uni_prob(0.5) and  gaussian_blur:
		augmented_images_set += np.hstack([
				[
					cv2.GaussianBlur(image, (i, i), 0)
					for i in xrange(1, gaussian_blur_config['kernel_size'], gaussian_blur_config['step_size'])
				]
				for image in images
			]).tolist()

	if uni_prob(0.5) and  median_blur:
		augmented_images_set += np.hstack([
				[
					cv2.medianBlur(image, i)
					for i in xrange(1, median_blur_config['kernel_size'], median_blur_config['step_size'])
				]
				for image in images
			]).tolist()

	if uni_prob(0.5) and  bilateral_blur:
		augmented_images_set += np.hstack([
				[
					cv2.bilateralFilter(image, i, i*2, i/2)
					for i in xrange(1, bilateral_blur_config['kernel_size'], bilateral_blur_config['step_size'])
				]
				for image in images
			]).tolist()

	if uni_prob(0.5) and  horizontal_flips:
		augmented_images_set += [cv2.flip(np.array(image), 1) for image in augmented_images_set]

	if uni_prob(0.5) and  shuffle_result:
		np.random.shuffle(augmented_images_set)

	return [np.array(image) for image in augmented_images_set]
  
