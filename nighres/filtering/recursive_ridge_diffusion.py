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
    input_image:

	ridge_intensities:

	ridge_filter:
	
	surface_levelset:

	orientation:

	ang_factor:

	loc_prior:
	
	nb_scales:

	propagation_model:

	diffusion_factor:

	similarity_scale:

	neighborhood_size:

	max_iter:

	max_diff:
    
    Returns
    ----------
   	dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * filter (niimg): 
        * proba (niimg): 
        * propagation (niimg): 
        * scale (niimg): 
        * ridge_direction (niimg): 
        * correction (niimg): 
		* ridge_size (niimg): 

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------

    """

    print('\n Recursive Ridge Diffusion')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        filter_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd_filter')

        proba_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd_proba')

        propagation_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='rrd_propag')

        scale_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='rrd_scale')

        ridge_direction_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd_dir')

        correction_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd_correct')
        
        ridge_size_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd_size')

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
    
    propagation_data = np.reshape(np.array(rrd.getPropagatedResponseImage(),
                                    dtype=np.float32), dimensions, 'F')
    
    scale_data = np.reshape(np.array(rrd.getDetectionScaleImage(),
                                   dtype=np.int32), dimensions, 'F')

    ridge_direction_data = np.reshape(np.array(rrd.getRidgeDirectionImage(),
                                    dtype=np.float32), (dimensions[0],dimensions[1],dimensions[2],3) , 'F')
    
    correction_data = np.reshape(np.array(rrd.getDirectionalCorrectionImage(),
                                    dtype=np.float32), dimensions, 'F')
    
    ridge_size_data = np.reshape(np.array(rrd.getRidgeSizeImage(),
                                    dtype=np.float32), dimensions, 'F')
    

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(filter_data)
    filter = nb.Nifti1Image(filter_data, affine, header)

    header['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, affine, header)

    header['cal_max'] = np.nanmax(propagation_data)
    propagation = nb.Nifti1Image(propagation_data, affine, header)

    header['cal_max'] = np.nanmax(scale_data)
    scale = nb.Nifti1Image(scale_data, affine, header)
    
    header['cal_max'] = np.nanmax(ridge_direction_data)
    ridge_direction = nb.Nifti1Image(ridge_direction_data, affine, header)
    
    header['cal_max'] = np.nanmax(correction_data)
    correction = nb.Nifti1Image(correction_data, affine, header)
    
    header['cal_max'] = np.nanmax(ridge_size_data)
    ridge_size = nb.Nifti1Image(ridge_size_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, filter_file), filter)
        save_volume(os.path.join(output_dir, proba_file), proba)
        save_volume(os.path.join(output_dir, propagation_file), propagation)
        save_volume(os.path.join(output_dir, scale_file), scale)
        save_volume(os.path.join(output_dir, ridge_direction_file), ridge_direction)
        save_volume(os.path.join(output_dir, correction_file), correction)
        save_volume(os.path.join(output_dir, ridge_size_file), ridge_size)

    return {'filter': filter, 'proba': proba,
            'propagation': propagation, 'scale': scale,
            'ridge_direction': ridge_direction, 'correction': correction, 'ridge_size': ridge_size}
