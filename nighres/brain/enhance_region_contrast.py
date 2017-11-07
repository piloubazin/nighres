import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from nighres.io import load_volume, save_volume
from nighres.utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_atlas_file
from colorama.ansi import Back


def enhance_region_contrast(intensity_image, segmentation_image,
                            levelset_boundary_image, atlas_file,
                            enhanced_region, contrast_background,
                            partial_voluming_distance,
                            save_data=False, output_dir=None,
                            file_name=None):
    
    """ Enhance Region Contrast

    Estimates brain structures from an atlas for MRI data using
    a Multiple Object Geometric Deformable Model (MGDM)

    Parameters
    ----------
    intensity_image: niimg
    
    segmentationImag : niimg
    
    levelset_boundary_image: niimg
    
    atlas_file: str, optional
        Path to plain text atlas file (default is stored in DEFAULT_ATLAS)
    
    enhanced_region: str
    
    contrast_background: str
    
    partial_voluming_distance: float
    

    Returns
    ----------
    

    Notes
    ----------
   

    References
    ----------
    """

    print('\n Enhance Region Contrast')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

        reg_file = _fname_4saving(file_name=file_name,
                                  rootfile=intensity_image,
                                  suffix='erc_reg')

        back_file = _fname_4saving(file_name=file_name,
                                  rootfile=intensity_image,
                                  suffix='erc_back')

        regProb_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='erc_regProb')

        backProb_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='erc_backProb')
        
        regPartVol_file = _fname_4saving(file_name=file_name,
                                   rootfile=intensity_image,
                                   suffix='erc_regPartVol')
        

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
   
    
    # reshape output to what nibabel likes
    reg_data = np.reshape(np.array(erc.getRegionMask(),
                                   dtype=np.int32), dimensions, 'F')

    back_data = np.reshape(np.array(erc.getBackgroundMask(),
                                    dtype=np.int32), dimensions, 'F')
    
    regProb_data = np.reshape(np.array(erc.getRegionProbability(),
                                   dtype=np.float32), dimensions, 'F')

    backProb_data = np.reshape(np.array(erc.getBackgroundProbability(),
                                    dtype=np.float32), dimensions, 'F')
    
    regPartVol_data = np.reshape(np.array(erc.getRegionPartialVolume(),
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

    header['cal_max'] = np.nanmax(regProb_data)
    regProb = nb.Nifti1Image(regProb_data, affine, header)

    header['cal_max'] = np.nanmax(backProb_data)
    backProb = nb.Nifti1Image(backProb_data, affine, header)
    
    header['cal_max'] = np.nanmax(regPartVol_data)
    regPartVol = nb.Nifti1Image(regPartVol_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, reg_file), reg)
        save_volume(os.path.join(output_dir, back_file), back)
        save_volume(os.path.join(output_dir, regProb_file), regProb)
        save_volume(os.path.join(output_dir, backProb_file), backProb)
        save_volume(os.path.join(output_dir, regPartVol_file), regPartVol)

    return {'region_mask': reg, 'background_mask': back,
            'region_prob': regProb, 'background_prob': backProb,
            'region_partial_vol': regPartVol}
