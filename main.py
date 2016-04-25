import astra
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plugins_astra
from image_operators import *
import time

def create_phantom_flowers(sx, sy, elem1, elem2):
    """
    Create phantom, share is two flowers (with 3 and 6 petals)
    """
    phantom = np.zeros((sx, sy))
    for x in range(sx):
        for y in range(sy):
            xx = x - 128.0
            yy = y - 128.0
            r = np.sqrt(xx*xx + yy*yy)
            if xx == 0:
                tetta = 0
            else:
                tetta = np.arctan(yy/xx)
                if xx < 0:
                    tetta += np.pi
            if r <= 50*(1 + np.cos(3*tetta) + pow(np.sin(3*tetta), 2)):
                phantom[x, y] = elem1
                if r <= 120*np.sin(6*tetta):
                    phantom[x, y] = 0.6*elem1 + 0.4*elem2
            else:
                if r <= 120*np.sin(6*tetta):
                    phantom[x, y] = elem2

    return phantom

def save_one_image(image1, title1, name, palit=plt.cm.gray, xlabel='', ylabel=''):
    f = plt.figure()

    ax = f.add_subplot(111)
    im1 = ax.imshow(image1, cmap=palit, interpolation='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title1)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    f.colorbar(im1, ax=ax, shrink=0.9)

    plt.savefig(name)
    return

def calculate_error(proj_id, img, data):
    W = astra.OpTomo(proj_id)
    sino = W * img
    sino = sino.reshape(data.shape) - data
    tv = norm1(gradient(sino))
    l2 = norm2sq(sino) 
    return tv, l2 

def cp_alg(proj_id, sid, vid):
    # CP algorithm
    cfg = astra.astra_dict('CP')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = vid
    cfg['option'] = {}
    cfg['option']['its_PM'] = 150
    cfg['option']['Lambda'] = 10000.0
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 100)
    rec = astra.data2d.get(vid)
    rec = np.flipud(rec)
    rec = rec.astype('float32')
    plt.imsave('rec_cp.png', rec, cmap = plt.cm.gray)
    save_one_image(rec, 'reconstruct', 'rec_cp_.png')

    astra.algorithm.delete(alg_id)
    return rec

def run_all_experiment(proj_id, sid, vid, sino):
    cfg = astra.astra_dict('CP')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = vid
    cfg['option'] = {}
    cfg['option']['its_PM'] = 100

    for i in np.arange(1.0, 1.01, 0.1):
        cfg['option']['Lambda'] = i
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 100)
        rec = astra.data2d.get(vid)
        rec = np.flipud(rec)
        rec = rec.astype('float32')
        tv, l2 = calculate_error(proj_id, rec, sino)
        print 'tv rec: ', tv, 'l2 norm rec: ', l2
        astra.algorithm.delete(alg_id)
    
        plt.imsave('rec_cp_Lamda_' + str(i) + '.png', rec, cmap = plt.cm.gray)
        save_one_image(rec, 'reconstruct', 'rec_cp_Lamda_' + str(i) + '_.png')
    return

def cg_alg(proj_id, sid, vid):
    # CG algorithm
    cfg = astra.astra_dict('CG')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = vid
    cfg['option'] = {}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 100)
    rec = astra.data2d.get(vid)
    rec = np.flipud(rec)
    plt.imsave('rec_cg.png', rec, cmap = plt.cm.gray)
    save_one_image(rec, 'reconstruct', 'rec_cg_.png')

    astra.algorithm.delete(alg_id)
    return rec

if __name__=='__main__':
    '''
    ph = create_phantom_flowers(256, 256, 250, 120)
    detector_cell = 256
    n_angles = 180
    save_one_image(ph, 'ph', 'ph.png')

    vol_geom = astra.create_vol_geom(detector_cell, detector_cell)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_cell, np.linspace(0,np.pi,n_angles,False))
    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    W = astra.OpTomo(proj_id)
    sinogram = W * ph
    sinogram = sinogram.reshape([n_angles, detector_cell])
    save_one_image(sinogram, 'sinogram', 'sinogram.png')
    '''

    #reconstruct nRecon
    # ----------
    input_rec = '/diskmnt/a/makov/yaivan/MMC_1/_tmp/astra/bh_92_rc_15/MMC1_2.82um__rec0960_astra_sart.png'
    original = plt.imread(input_rec)
    original = original.astype('float32')
    if len(original.shape) == 3:
        original = original[...,0]
    max_v = original.max()
    max_ = 273.353
    min_ = -100.677
    original = (original / max_v) * (max_ + min_) - min_

    # Sinogram
    # ----------
    path_sino = '/diskmnt/a/makov/yaivan/MMC_1/_tmp/nrecon/bh_92_rc_15/MMC1_2.82um__sino0960.tif'
    sinogram = plt.imread(path_sino)
    if len(sinogram.shape) == 3:
        sinogram = sinogram[...,0]
    sinogram = np.flipud(sinogram)
    sinogram = sinogram[0:1800,:]
    detector_cell = sinogram.shape[1] #number of detector cells
    n_angles = sinogram.shape[0] #number of proj. angles

    pixel_size = 2.82e-3
    os_distance = 56.135/pixel_size
    ds_distance = 225.082/pixel_size

    angles_reduce = 1
    angles = np.arange(n_angles)*0.1*angles_reduce
    angles = angles.astype('float32')/180*np.pi
    angles = angles - (angles.max()+angles.min())/2
    angles = angles + np.pi/2

    vol_geom = astra.create_vol_geom(detector_cell, detector_cell)
    proj_geom = astra.create_proj_geom('fanflat', ds_distance / os_distance, detector_cell, angles,
                                            os_distance, (ds_distance - os_distance))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom) #line_fanflat

    #Calculate tv and l2 norm for original
    tv, l2 = calculate_error(proj_id, original, sinogram)
    print 'tv SART: ', tv, 'l2 norm SART: ', l2

    # Then, we register the plugin class with ASTRA
    astra.plugin.register(plugins_astra.CP_Plugin)
    astra.plugin.register(plugins_astra.CG_Plugin)

    # Create data structures
    sid = astra.data2d.create('-sino', proj_geom, sinogram)
    vid = astra.data2d.create('-vol', vol_geom)

    #rec = cp_alg(proj_id, sid, vid)
    #rec = cg_alg(proj_id, sid, vid)

    #print 'parametr for CG'
    #l2, tv = calculate_error(rec)

    run_all_experiment(proj_id, sid, vid, sinogram)

    # Clean up
    astra.projector.delete(proj_id)
    astra.data2d.delete([vid, sid])

    #t1 = time.time()
    #print t1 - t0
