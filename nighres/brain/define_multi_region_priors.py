import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from nighres.io import load_volume, save_volume
from nighres.utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_atlas_file
from colorama.ansi import Back

def define_multi_region_priors(segmentation_image,levelset_boundary_image,
                               atlas_file, #defined_region,
                               definition_method, distance_offset,
                               save_data=False, output_dir=None,
                               file_name=None):
    
    """ Define Multi-Region Priors

    Enhances the contrast between selected regions from a MGDM brain segmentation.

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

    print('\n Define Multi-Region Priors')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, segmentation_image)

        intervent_file = _fname_4saving(file_name=file_name,
                                  rootfile=segmentation_image,
                                  suffix='dmrp_intervent')

        horns_file = _fname_4saving(file_name=file_name,
                                  rootfile=segmentation_image,
                                  suffix='dmrp_horns')
        
        intercap_file = _fname_4saving(file_name=file_name,
                                  rootfile=segmentation_image,
                                  suffix='dmrp_intercap')    

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create DefineMultiRegionPriors instance

    dmrp = cbstools.BrainDefineMultiRegionPriors()
 
    # set erc parameters
    dmrp.setAtlasFile(atlas_file)
    #dmrp.setDefinedRegion(defined_region)
    dmrp.setDefinitionMethod(definition_method)
    dmrp.setDistanceOffset(distance_offset)

    # load segmentation image and use it to set dimensions and resolution
    img = load_volume(segmentation_image)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    dmrp.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    dmrp.setResolutions(resolution[0], resolution[1], resolution[2])
    
    # input segmentation_image
    dmrp.setSegmentationImage(cbstools.JArray('int')((data.flatten('F')).astype(int)))

    # input levelset_boundary_image
    data = load_volume(levelset_boundary_image).get_data()
    dmrp.setLevelsetBoundaryImage(cbstools.JArray('float')((data.flatten('F')).astype(float)))

    # execute DMRP
    try:
        dmrp.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return
   
    
    # reshape output to what nibabel likes
    intervent_data = np.reshape(np.array(dmrp.getInterVentricularPV(),
                                   dtype=np.float32), dimensions, 'F')

    horns_data = np.reshape(np.array(dmrp.getVentricularHornsPV(),
                                   dtype=np.float32), dimensions, 'F')
    
    intercap_data = np.reshape(np.array(dmrp.getInternalCapsulePV(),
                                   dtype=np.float32), dimensions, 'F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(intervent_data)
    intervent = nb.Nifti1Image(intervent_data, affine, header)

    header['cal_max'] = np.nanmax(horns_data)
    horns = nb.Nifti1Image(horns_data, affine, header)
    
    header['cal_max'] = np.nanmax(intercap_data)
    intercap = nb.Nifti1Image(intercap_data, affine, header)

   
    if save_data:
        save_volume(os.path.join(output_dir, intervent_file), intervent)
        save_volume(os.path.join(output_dir, horns_file), horns)
        save_volume(os.path.join(output_dir, intercap_file), intercap)


    return {'intervent': intervent, 'horns': horns, 'intercap': intercap}
