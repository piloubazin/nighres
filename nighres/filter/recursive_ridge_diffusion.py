import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from nighres.io import load_volume, save_volume
from nighres.utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_atlas_file


def recursive_ridge_diffusion(input_image, ridge_intensities, ridge_filter,
                              surface_levelset,orientation, ang_factor, loc_prior, 
                              nb_scales, propagation_model, diffusion_factor, 
                              similarity_scale, neighborhood_size,
                              max_iter,max_diff,
                              save_data=False, output_dir=None,
                              file_name=None):
    
    """ Recursive Ridge Diffusion

    Extracts planar of tubular structures across multiple scales, with an optional directional bias.

    Parameters
    ----------
    
    
    Returns
    ----------
   

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Algorithm details can be
    found in [1]_ and [2]_

    References
    ----------
    .. [1] Bogovic, Prince and Bazin (2013). A multiple object geometric
       deformable model for image segmentation.
       doi:10.1016/j.cviu.2012.10.006.A
    .. [2] Fan, Bazin and Prince (2008). A multi-compartment segmentation
       framework with homeomorphic level sets. DOI: 10.1109/CVPR.2008.4587475
    """

    print('\n Recursive Ridge Diffusion')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        filter_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='ridge_filter')

        proba_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='ridge_proba')

        propag_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='ridge_propag')

        scale_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='ridge_scale')

        direction_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='ridge_direction')

        correct_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='ridge_correct')
        
        size_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='ridge_size')

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create extraction instance
    rrd = cbstools.FilterRecursiveRidgeDiffusion()

    # set parameters
    rrd.setRidgeIntensities(ridge_intensities)
    rrd.setRidgeFilter(ridge_filter)
    rrd.setOrientationToSurface(orientation)
    rrd.setAngularFactor(ang_factor)
    rrd.setNumberOfScales(nb_scales)
    rrd.setPropagationModel(propagation_model)
    rrd.setDiffusionFactor(diffusion_factor)
    rrd.setSimilarityScale(similarity_scale)
    rrd.setNeighborhoodSize(neighborhood_size)
    rrd.setMaxIterations(max_iter)
    rrd.setMaxDifference(max_diff)
                     
                                                   
    # load input image and use it to set dimensions and resolution
    img = load_volume(input_image)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    rrd.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    rrd.setResolutions(resolution[0], resolution[1], resolution[2])

    # input input_image
    rrd.setInputImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))

    # input surface_levelset
    data = load_volume(surface_levelset).get_data()
    rrd.setSurfaceLevelSet(cbstools.JArray('float')((data.flatten('F')).astype(float)))
    
    # input location prior image
    data = load_volume(loc_prior).get_data()
    rrd.setLocationPrior(cbstools.JArray('float')((data.flatten('F')).astype(float)))
    
    
    # execute Extraction
    try:
        rrd.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # reshape output to what nibabel likes
    filter_data = np.reshape(np.array(rrd.getFilterResponseImage(),
                                   dtype=np.float32), dimensions, 'F')

    proba_data = np.reshape(np.array(rrd.getProbabilityResponseImage(),
                                    dtype=np.float32), dimensions, 'F')
    
    propag_data = np.reshape(np.array(rrd.getPropagatedResponseImage(),
                                    dtype=np.float32), dimensions, 'F')
    
    scale_data = np.reshape(np.array(rrd.getDetectionScaleImage(),
                                   dtype=np.int32), dimensions, 'F')

    direction_data = np.reshape(np.array(rrd.getRidgeDirectionImage(),
                                    dtype=np.float32), (dimensions[0],dimensions[1],dimensions[2],3) , 'F')
    
    correct_data = np.reshape(np.array(rrd.getDirectionalCorrectionImage(),
                                    dtype=np.float32), dimensions, 'F')
    
    size_data = np.reshape(np.array(rrd.getRidgeSizeImage(),
                                    dtype=np.float32), dimensions, 'F')
    

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(filter_data)
    filter = nb.Nifti1Image(filter_data, affine, header)

    header['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, affine, header)

    header['cal_max'] = np.nanmax(propag_data)
    propag = nb.Nifti1Image(propag_data, affine, header)

    header['cal_max'] = np.nanmax(scale_data)
    scale = nb.Nifti1Image(scale_data, affine, header)
    
    header['cal_max'] = np.nanmax(direction_data)
    direction = nb.Nifti1Image(direction_data, affine, header)
    
    header['cal_max'] = np.nanmax(correct_data)
    correct = nb.Nifti1Image(correct_data, affine, header)
    
    header['cal_max'] = np.nanmax(size_data)
    size = nb.Nifti1Image(size_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, filter_file), filter)
        save_volume(os.path.join(output_dir, proba_file), proba)
        save_volume(os.path.join(output_dir, propag_file), propag)
        save_volume(os.path.join(output_dir, scale_file), scale)
        save_volume(os.path.join(output_dir, direction_file), direction)
        save_volume(os.path.join(output_dir, correct_file), correct)
        save_volume(os.path.join(output_dir, size_file), size)

    return {'filter': filter, 'proba': proba,
            'propag': propag, 'scale': scale,
            'direction': direction, 'correct': correct, 'size': size}
