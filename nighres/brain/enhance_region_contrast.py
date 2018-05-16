import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_atlas_file
from colorama.ansi import Back


def enhance_region_contrast(intensity_image, segmentation_image,
                            levelset_boundary_image, atlas_file,
                            enhanced_region, contrast_background,
                            partial_voluming_distance,
                            save_data=False, output_dir=None,
                            file_name=None):
    
    """ Enhance Region Contrast

    Enhances the contrast between selected regions from a MGDM brain segmentation.
    
    Parameters
    ----------
    intensity_image: niimg
        Intensity contrast to enhance between the chosen regions
    
    segmentation_image : niimg
       MGDM brain segmentation image (_mgdm_seg)
    
    levelset_boundary_image: niimg
       MGDM distance to closest boundary (_mgdm_dist)
    
    atlas_file: str, optional
        Path to MGDM brain atlas file (default is stored in DEFAULT_ATLAS)
    
    enhanced_region: str
       Region of interest to enhance (choices are: 'crwm', 'cbwm', 'csf' for
       cerebral and cerebellar WM, CSF)
       
    contrast_background: str
      Region to contrast as background (choices are: 'crgm', 'crwm', 'brain'
      for cerebral and cerebellar GM, brain tissues)
      
    partial_voluming_distance: float
      Distance in voxels for estimating partial voluming at the boundaries

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets, with # the region and % the background label above)

        * region_mask (niimg): Hard segmentation mask of the (GM) region
          of interest (_emask_#)
        * background_mask (niimg): Hard segmentation mask of the (CSF) region
          background (_emask_%)
        * region_proba (niimg): Probability map of the (GM) region
          of interest (_eproba_#)
        * background_proba (niimg): Probability map of the (CSF) region
          background (_eproba_%)
        * region_pv (niimg): Levelset surface of the (GM) region
          of interest (_epv_#)
        * background_pv (niimg): Levelset surface of the (CSF) region
          background (_epv_%)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------
    """

    print('\n Enhance Region Contrast')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create EnhanceRegionContrast instance
    erc = cbstools.BrainEnhanceRegionContrast()

    # set erc parameters
    erc.setAtlasFile(atlas_file)
    erc.setEnhancedRegion(enhanced_region)
    erc.setContrastBackground(contrast_background)
    erc.setPartialVolumingDistance(partial_voluming_distance)
    erc.setComponents(3) # not used in module

    # load intensity_image and use it to set dimensions and resolution
    img = load_volume(intensity_image)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    erc.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    erc.setResolutions(resolution[0], resolution[1], resolution[2])
    
    # input intensity_image
    erc.setIntensityImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))

    # input segmentation_image
    data = load_volume(segmentation_image).get_data()
    erc.setSegmentationImage(cbstools.JArray('int')((data.flatten('F')).astype(int)))

    # input levelset_boundary_image
    data = load_volume(levelset_boundary_image).get_data()
    erc.setLevelsetBoundaryImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))

    # execute ERC
    try:
        erc.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return
   

    if save_data:
        reg_file = _fname_4saving(file_name=file_name,
                                  rootfile=intensity_image,
                                  suffix='emask'+str(erc.getRegionName()))

        back_file = _fname_4saving(file_name=file_name,
                                  rootfile=intensity_image,
                                  suffix='emask'+str(erc.getBackgroundName()))

        reg_proba_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='eproba'+str(erc.getRegionName()))

        back_proba_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='eproba'+str(erc.getBackgroundName()))
        
        reg_pv_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='epv'+str(erc.getRegionName()))
        
        back_pv_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='epv'+str(erc.getBackgroundName()))
        
    
    # reshape output to what nibabel likes
    reg_data = np.reshape(np.array(erc.getRegionMask(),
                                   dtype=np.int32), dimensions, 'F')

    back_data = np.reshape(np.array(erc.getBackgroundMask(),
                                    dtype=np.int32), dimensions, 'F')
    
    reg_proba_data = np.reshape(np.array(erc.getRegionProbability(),
                                   dtype=np.float32), dimensions, 'F')

    back_proba_data = np.reshape(np.array(erc.getBackgroundProbability(),
                                    dtype=np.float32), dimensions, 'F')
    
    reg_pv_data = np.reshape(np.array(erc.getRegionPartialVolume(),
                                    dtype=np.float32), dimensions, 'F')
    
    back_pv_data = np.reshape(np.array(erc.getBackgroundPartialVolume(),
                                    dtype=np.float32), dimensions, 'F')

    ## membership and labels output has a 4th dimension, set to 6
    #dimensions4d = [dimensions[0], dimensions[1], dimensions[2], 6]
    #lbl_data = np.reshape(np.array(mgdm.getPosteriorMaximumLabels4D(),
    #                               dtype=np.int32), dimensions4d, 'F')
    #mems_data = np.reshape(np.array(mgdm.getPosteriorMaximumMemberships4D(),
    #                                dtype=np.float32), dimensions4d, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(reg_data)
    reg = nb.Nifti1Image(reg_data, affine, header)

    header['cal_max'] = np.nanmax(back_data)
    back = nb.Nifti1Image(back_data, affine, header)

    header['cal_max'] = np.nanmax(reg_proba_data)
    reg_proba = nb.Nifti1Image(reg_proba_data, affine, header)

    header['cal_max'] = np.nanmax(back_proba_data)
    back_proba = nb.Nifti1Image(back_proba_data, affine, header)
    
    header['cal_max'] = np.nanmax(reg_pv_data)
    reg_pv = nb.Nifti1Image(reg_pv_data, affine, header)

    header['cal_max'] = np.nanmax(back_pv_data)
    back_pv = nb.Nifti1Image(back_pv_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, reg_file), reg)
        save_volume(os.path.join(output_dir, back_file), back)
        save_volume(os.path.join(output_dir, reg_proba_file), reg_proba)
        save_volume(os.path.join(output_dir, back_proba_file), back_proba)
        save_volume(os.path.join(output_dir, reg_pv_file), reg_pv)
        save_volume(os.path.join(output_dir, back_pv_file), back_pv)

    return {'region_mask': reg, 'background_mask': back,
            'region_proba': reg_proba, 'background_proba': back_proba,
            'region_pv': reg_pv, 'background_pv': back_pv}
