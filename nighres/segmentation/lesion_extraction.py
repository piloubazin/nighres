import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from nighres.io import load_volume, save_volume
from nighres.utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_atlas_file


def lesion_extraction(probability_image, segmentation_image,
                      levelset_boundary_image, location_prior_image,
                      atlas_file,
                      gm_boundary_partial_vol_dist, csf_boundary_partial_vol_dist,
                      lesion_clust_dist, prob_min_thresh, prob_max_thresh,
                      small_lesion_size,
                      save_data=False, output_dir=None,
                      file_name=None):
    
    """ Lesion Extraction

    Extracts lesions from a probability image and a pre-segmentation with MGDM.

    Parameters
    ----------
    probability_image: niimg

	segmentation_image: niimg

    levelset_boundary_image: niimg
        MGDM distance to closest boundary (_mgdm_dist)

	location_prior_image: niimg
	   
	atlas_file: str
	    Path to MGDM brain atlas file (default is stored in DEFAULT_ATLAS)

	gm_boundary_partial_vol_dist: float

	csf_boundary_partial_vol_dist: float

	lesion_clust_dist: float

	prob_min_tresh: float

	prob_max_tresh: float

	small_lesion_size: float

    
    Returns
    ----------
   	dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * lesion_prior (niimg): 
        * lesion_size (niimg): 
        * lesion_proba (niimg): 
        * lesion_pv (niimg): 
        * lesion_labels (niimg): 
        * lesion_score (niimg): 

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. 

    References
    ----------

    """

    print('\n Lesion Extraction')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, probability_image)

        lesion_prior_file = _fname_4saving(file_name=file_name,
                                  rootfile=probability_image,
                                  suffix='lesion_prior')

        lesion_size_file = _fname_4saving(file_name=file_name,
                                  rootfile=probability_image,
                                  suffix='lesion_size')

        lesion_proba_file = _fname_4saving(file_name=file_name,
                                   rootfile=probability_image,
                                   suffix='lesion_proba')

        lesion_pv_file = _fname_4saving(file_name=file_name,
                                   rootfile=probability_image,
                                   suffix='lesion_pv')

        lesion_labels_file = _fname_4saving(file_name=file_name,
                                   rootfile=probability_image,
                                   suffix='lesion_labels')

        lesion_score_file = _fname_4saving(file_name=file_name,
                                   rootfile=probability_image,
                                   suffix='lesion_score')

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create extraction instance
    el = cbstools.SegmentationLesionExtraction()

    # set extraction parameters
    el.setComponents(3) # not used in module
    el.setAtlasFile(atlas_file)
    el.setGMPartialVolumingDistance(gm_boundary_partial_vol_dist)
    el.setCSFPartialVolumingDistance(csf_boundary_partial_vol_dist)
    el.setLesionClusteringDistance(lesion_clust_dist)
    el.setMinProbabilityThreshold(prob_min_thresh)
    el.setMaxProbabilityThreshold(prob_max_thresh)
    el.setMinimumSize(small_lesion_size)
    

    # load segmentation image and use it to set dimensions and resolution
    img = load_volume(segmentation_image)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    el.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    el.setResolutions(resolution[0], resolution[1], resolution[2])

    # input segmentation_image
    el.setSegmentationImage(cbstools.JArray('int')((data.flatten('F')).astype(int)))

    # input levelset_boundary_image
    data = load_volume(levelset_boundary_image).get_data()
    el.setLevelsetBoundaryImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))
    
    # input levelset_boundary_image
    data = load_volume(probability_image).get_data()
    el.setProbaImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))
    
    # input levelset_boundary_image
    data = load_volume(location_prior_image).get_data()
    el.setLocationPriorImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))
    

    # execute Extraction
    try:
        el.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # reshape output to what nibabel likes
    lesion_prior_data = np.reshape(np.array(el.getRegionPrior(),
                                   dtype=np.float32), dimensions, 'F')

    lesion_size_data = np.reshape(np.array(el.getLesionSize(),
                                    dtype=np.float32), dimensions, 'F')
    
    lesion_proba_data = np.reshape(np.array(el.getLesionProba(),
                                    dtype=np.float32), dimensions, 'F')
    
    lesion_pv_data = np.reshape(np.array(el.getBoundaryPartialVolume(),
                                   dtype=np.float32), dimensions, 'F')

    lesion_labels_data = np.reshape(np.array(el.getLesionLabels(),
                                    dtype=np.int32), dimensions, 'F')
    
    lesion_score_data = np.reshape(np.array(el.getLesionScore(),
                                    dtype=np.float32), dimensions, 'F')
    

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(lesion_prior_data)
    lesion_prior = nb.Nifti1Image(lesion_prior_data, affine, header)

    header['cal_max'] = np.nanmax(lesion_size_data)
    lesion_size = nb.Nifti1Image(lesion_size_data, affine, header)

    header['cal_max'] = np.nanmax(lesion_proba_data)
    lesion_proba = nb.Nifti1Image(lesion_proba_data, affine, header)

    header['cal_max'] = np.nanmax(lesion_pv_data)
    lesion_pv = nb.Nifti1Image(lesion_pv_data, affine, header)
    
    header['cal_max'] = np.nanmax(lesion_labels_data)
    lesion_labels = nb.Nifti1Image(lesion_labels_data, affine, header)
    
    header['cal_max'] = np.nanmax(lesion_score_data)
    lesion_score = nb.Nifti1Image(lesion_score_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, lesion_prior_file), lesion_prior)
        save_volume(os.path.join(output_dir, lesion_size_file), lesion_size)
        save_volume(os.path.join(output_dir, lesion_proba_file), lesion_proba)
        save_volume(os.path.join(output_dir, lesion_pv_file), lesion_pv)
        save_volume(os.path.join(output_dir, lesion_labels_file), lesion_labels)
        save_volume(os.path.join(output_dir, lesion_score_file), lesion_score)

    return {'lesion_prior': lesion_prior, 'lesion_size': lesion_size,
            'lesion_proba': lesion_proba, 'lesion_pv': lesion_pv,
            'lesion_labels': lesion_labels, 'lesion_score': lesion_score}
