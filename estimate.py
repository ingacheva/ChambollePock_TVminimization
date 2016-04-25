import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_operators import *

def calculate_error(image):
    max_v = image.max()
    min_v = image.min()
    image = (image - min_v)/ (max_v - min_v)
    tv = norm1(image)
    l2 = norm2sq(image)
    print 'l2 norm: ', l2
    print 'TV: ', tv
    return l2, tv

if __name__=='__main__':

	input_rec = '/diskmnt/a/makov/yaivan/MMC_1/_tmp/astra/bh_92_rc_15/MMC1_2.82um__rec0960_astra_sart.png'
	original1 = plt.imread(input_rec)
	original1 = original1[...,0]
	original1 = original1.astype('float32')
	print 'parametrs for SIRT'
	l2, tv = calculate_error(original1)

	input_rec = '/diskmnt/a/makov/yaivan/MMC_1/_tmp/nrecon/bh_92_rc_15/MMC1_2.82um__rec0960.png'
	original2 = plt.imread(input_rec)
	original2 = original2[...,0]
	original2 = original2.astype('float32')
	print 'parametrs for nRecon'
	l2, tv = calculate_error(original2)

	input_rec = '/home/ingacheva/_tomo/Plugin_ChambollePork/rec_cp_Lamda_1.0.png'
	rec1 = plt.imread(input_rec)
	rec1 = rec1[...,0]
	rec1 = rec1.astype('float32')
	print 'parametrs for CP l = 1'
	l2, tv = calculate_error(rec1)

	input_rec = '/home/ingacheva/_tomo/Plugin_ChambollePork/rec_cp_Lamda_1000.0.png'
	rec2 = plt.imread(input_rec)
	rec2 = rec2[...,0]
	rec2 = rec2.astype('float32')
	print 'parametrs for CP l = 1000'
	l2, tv = calculate_error(rec2)

	input_rec = '/home/ingacheva/_tomo/Plugin_ChambollePork/rec_cp_Lamda_1000000.0.png'
	rec3 = plt.imread(input_rec)
	rec3 = rec3[...,0]
	rec3 = rec3.astype('float32')
	print 'parametrs for CP l = 1000000'
	l2, tv = calculate_error(rec3)

	k = 2650
	t = range(original1.shape[0])
	plt.figure()
	plt.plot(original1[k, t], label="nRecon")
	plt.plot(rec1[k, t], label="CP L = 1")
	plt.plot(rec3[k, t], label="CP L = 1000000")
	plt.legend()
	plt.grid(True)
	plt.savefig('energy.png')
