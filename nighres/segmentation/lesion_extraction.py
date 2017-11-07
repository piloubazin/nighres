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
    probability_image, semgnetation_image,
                      levelset_boundary_image,location_prior_image,
                      atlas_file,
                      gm_boundary_partial_vol_dist, csf_boundary_partial_vol_dist,
                      lesion_clust_dist, prob_min_tresh, prob_max_tresh,
                      small_lesion_size,
                      save_data=False, output_dir=None,
                      file_name=None
    
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

    print('\n Lesion Extraction')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, probability_image)

        reg_file = _fname_4saving(file_name=file_name,
                                  rootfile=probability_image,
                                  suffix='extract_reg')

        legSize_file = _fname_4saving(file_name=file_name,
                                  rootfile=probability_image,
                                  suffix='extract_legSize')

        legProb_file = _fname_4saving(file_name=file_name,
                                   rootfile=probability_image,
                                   suffix='extract_legProb')

        bound_file = _fname_4saving(file_name=file_name,
                                   rootfile=probability_image,
                                   suffix='extract_bound')

        label_file = _fname_4saving(file_name=file_name,
                                  rootfile=probability_image,
                                  suffix='extract_label')

        score_file = _fname_4saving(file_name=file_name,
                                  rootfile=probability_image,
                                  suffix='extract_score')

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
    reg_data = np.reshape(np.array(el.getRegionPrior(),
                                   dtype=np.float32), dimensions, 'F')

    legSize_data = np.reshape(np.array(el.getLesionSize(),
                                    dtype=np.float32), dimensions, 'F')
    
    legProb_data = np.reshape(np.array(el.getLesionProba(),
                                    dtype=np.float32), dimensions, 'F')
    
    bound_data = np.reshape(np.array(el.getBoundaryPartialVolume(),
                                   dtype=np.float32), dimensions, 'F')

    label_data = np.reshape(np.array(el.getLesionLabels(),
                                    dtype=np.int32), dimensions, 'F')
    
    score_data = np.reshape(np.array(el.getLesionScore(),
                                    dtype=np.float32), dimensions, 'F')
    

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(reg_data)
    reg = nb.Nifti1Image(reg_data, affine, header)

    header['cal_max'] = np.nanmax(legSize_data)
    legSize = nb.Nifti1Image(legSize_data, affine, header)

    header['cal_max'] = np.nanmax(legProb_data)
    legProb = nb.Nifti1Image(legProb_data, affine, header)

    header['cal_max'] = np.nanmax(bound_data)
    bound = nb.Nifti1Image(bound_data, affine, header)
    
    header['cal_max'] = np.nanmax(label_data)
    label = nb.Nifti1Image(label_data, affine, header)
    
    header['cal_max'] = np.nanmax(score_data)
    score = nb.Nifti1Image(score_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, reg_file), reg)
        save_volume(os.path.join(output_dir, legSize_file), legSize)
        save_volume(os.path.join(output_dir, legProb_file), legProb)
        save_volume(os.path.join(output_dir, bound_file), bound)
        save_volume(os.path.join(output_dir, label_file), label)
        save_volume(os.path.join(output_dir, score_file), score)

    return {'region': reg, 'lesion_size': legSize,
            'lesion_proba': legProb, 'boundary': bound,
            'label': label, 'score': score}
