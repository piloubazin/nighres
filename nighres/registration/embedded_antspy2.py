# basic dependencies
import os
import sys
from glob import glob
import math

# main dependencies: numpy, nibabel, ants
import numpy
import nibabel
import ants.utils
import ants

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir
from ..surface import probability_to_levelset
from ..shape import levelset_thickness

# convenience labels
X=0
Y=1
Z=2
T=3

def embedded_antspy2(source_image, target_image,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTSpy Registration

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines.

    Parameters
    ----------
    source_image: niimg
        Image to register
    target_image: niimg
        Reference image to match
    run_rigid: bool
        Whether or not to run a rigid registration first (default is False)
    rigid_iterations: float
        Number of iterations in the rigid step (default is 1000)
    run_similarity: bool
        Whether or not to run a similarity (rigid+scale) registration first (default is False)
    similarity_iterations: float
        Number of iterations in the similarity step (default is 1000)
    run_affine: bool
        Whether or not to run a affine registration first (default is False)
    affine_iterations: float
        Number of iterations in the affine step (default is 1000)
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
    regularization: {'Low', 'Medium', 'High'}
        Regularization preset for the SyN deformation (default is 'Medium')
    convergence: float
        Threshold for convergence, can make the algorithm very slow
        (default is convergence)
    mask_zero: bool
        Mask regions with zero value using ANTs masking option (default is False)
    smooth_mask: float
        Smoothly mask regions within a given ratio of the object's thickness,
        in [0.0, 1.0] (default is 0.0). This does not use ANTs masking.
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_header: bool
        Ignore the orientation information and affine matrix information
        extracted from the image header (default is False)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * transformed_source (niimg): Deformed source image (_ants-def)
        * mapping (niimg): Coordinate mapping from source to target (_ants-map)
        * inverse (niimg): Inverse coordinate mapping from target to source
          (_ants-invmap)

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. The
    interfacing with ANTs is performed through Nipype [2]_. Parameters have
    been set to values commonly found in neuroimaging scripts online, but not
    necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    .. [2] Gorgolewski et al (2011) Nipype: a flexible, lightweight and
       extensible neuroimaging data processing framework in python. Front
       Neuroinform 5. doi:10.3389/fninf.2011.00013
    """

    # just overloading the multi-channel version
    return embedded_antspy2_multi([source_image], [target_image],
                    run_rigid, rigid_iterations, run_similarity, similarity_iterations,
                    run_affine, affine_iterations,
                    run_syn, coarse_iterations, medium_iterations, fine_iterations,
					scaling_factor, cost_function, interpolation, regularization, 
					convergence, mask_zero, smooth_mask, ignore_affine, ignore_header,
					save_data, overwrite, output_dir, file_name)


def embedded_antspy2_2d(source_image, target_image,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
                    scaling_factor=32,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,
					ignore_affine=False, ignore_orient=False, ignore_res=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTSpy Registration 2D

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines.

    Parameters
    ----------
    source_image: niimg
        Image to register
    target_image: niimg
        Reference image to match
    run_rigid: bool
        Whether or not to run a rigid registration first (default is False)
    rigid_iterations: float
        Number of iterations in the rigid step (default is 1000)
    run_similarity: bool
        Whether or not to run a similarity (rigid+scale) registration first (default is False)
    similarity_iterations: float
        Number of iterations in the similarity step (default is 1000)
    run_affine: bool
        Whether or not to run a affine registration first (default is False)
    affine_iterations: float
        Number of iterations in the affine step (default is 1000)
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
    convergence: flaot
        Threshold for convergence, can make the algorithm very slow
        (default is convergence)
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_header: bool
        Ignore the orientation information and affine matrix information
        extracted from the image header (default is False)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * transformed_source (niimg): Deformed source image (_ants-def)
        * mapping (niimg): Coordinate mapping from source to target (_ants-map)
        * inverse (niimg): Inverse coordinate mapping from target to source
          (_ants-invmap)

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. Parameters
    have been set to values commonly found in neuroimaging scripts online, but
    not necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    """

    return embedded_antspy2_2d_multi([source_image], [target_image], 
                    None,
                    run_rigid, rigid_iterations,
                    run_similarity, similarity_iterations,
                    run_affine, affine_iterations,
                    run_syn, coarse_iterations, medium_iterations, 
                    fine_iterations, scaling_factor,
					cost_function,interpolation,regularization,
					convergence,mask_zero,
					ignore_affine, ignore_orient, ignore_res,
                    save_data, overwrite, output_dir,file_name)


def embedded_antspy2_2d_multi(source_images, target_images, image_weights=None,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
                    scaling_factor=32,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,
					ignore_affine=False, ignore_orient=False, ignore_res=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTSpy Registration 2D Multi-contrasts

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines. Uses all input contrasts
    with equal weights.

    Parameters
    ----------
    source_images: [niimg]
        Images to register
    target_images: [niimg]
        Reference images to match
    image_weights: [float]
        Relative weights to give each pair of images (default is equal)
    run_rigid: bool
        Whether or not to run a rigid registration first (default is False)
    rigid_iterations: float
        Number of iterations in the rigid step (default is 1000)
    run_similarity: bool
        Whether or not to run a similarity (rigid+scale) registration first (default is False)
    similarity_iterations: float
        Number of iterations in the similarity step (default is 1000)
    run_affine: bool
        Whether or not to run a affine registration first (default is False)
    affine_iterations: float
        Number of iterations in the affine step (default is 1000)
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
    convergence: flaot
        Threshold for convergence, can make the algorithm very slow
        (default is convergence)
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_orient: bool
        Ignore the orientation information and affine matrix information
        extracted from the image header (default is False)
    ignore_res: bool
        Ignore the resolution information extracted from the image header
        (default is False)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * transformed_source (niimg): Deformed source image (_ants-def)
        * mapping (niimg): Coordinate mapping from source to target (_ants-map)
        * inverse (niimg): Inverse coordinate mapping from target to source
          (_ants-invmap)

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. Parameters
    have been set to values commonly found in neuroimaging scripts online, but
    not necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    """

    print('\nEmbedded ANTs Registration 2D Multi-contrasts')

    # make sure that saving related parameters are correct

     # filenames needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_images[0])

    transformed_source_files = []
    for idx,source_image in enumerate(source_images):
        transformed_source_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-def'+str(idx))))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-invmap'))
    if save_data:
        if overwrite is False \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            missing = False
            for trans_file in transformed_source_files:
                if not os.path.isfile(trans_file):
                    missing = True

            if not missing:
                print("skip computation (use existing results)")
                transformed = []
                for trans_file in transformed_source_files:
                    transformed.append(trans_file)
                output = {'transformed_sources': transformed,
                      'transformed_source': transformed[0],
                      'mapping': mapping_file,
                      'inverse': inverse_mapping_file}
                return output


    # load and get dimensions and resolution from input images
    sources = []
    targets = []
    src_img_files = []
    trg_img_files = []
    for idx,img in enumerate(source_images):
        source = load_volume(source_images[idx])
        src_affine = source.affine
        src_header = source.header
        nsx = source.header.get_data_shape()[X]
        nsy = source.header.get_data_shape()[Y]
        nsz = 1
        rsx = source.header.get_zooms()[X]
        rsy = source.header.get_zooms()[Y]
        rsz = 1.0

        orig_src_aff = source.affine
        orig_src_hdr = source.header

        target = load_volume(target_images[idx])
        trg_affine = target.affine
        trg_header = target.header
        ntx = target.header.get_data_shape()[X]
        nty = target.header.get_data_shape()[Y]
        ntz = 1
        rtx = target.header.get_zooms()[X]
        rty = target.header.get_zooms()[Y]
        rtz = 1.0

        orig_trg_aff = target.affine
        orig_trg_hdr = target.header

        # in case the affine transformations are not to be trusted: make them equal
        if ignore_affine or ignore_orient or ignore_res:
            mx = numpy.argmax(numpy.abs(src_affine[0][0:3]))
            my = numpy.argmax(numpy.abs(src_affine[1][0:3]))
            mz = numpy.argmax(numpy.abs(src_affine[2][0:3]))
            new_affine = numpy.zeros((4,4))
            if ignore_res:
                new_affine[0][:] = src_affine[0][:]/rsx
                new_affine[1][:] = src_affine[1][:]/rsy
                new_affine[2][:] = src_affine[2][:]/rsz
                rsx = 1.0
                rsy = 1.0
                rsz = 1.0

            if ignore_orient:
                new_affine[0][0] = rsx
                new_affine[1][1] = rsy
                new_affine[2][2] = rsz
                new_affine[0][3] = -rsx*nsx/2.0
                new_affine[1][3] = -rsy*nsy/2.0
                new_affine[2][3] = -rsz*nsz/2.0
            elif ignore_affine:
                new_affine[0][mx] = rsx*numpy.sign(src_affine[0][mx])
                new_affine[1][my] = rsy*numpy.sign(src_affine[1][my])
                new_affine[2][mz] = rsz*numpy.sign(src_affine[2][mz])
                if (numpy.sign(src_affine[0][mx])<0):
                    new_affine[0][3] = rsx*nsx/2.0
                else:
                    new_affine[0][3] = -rsx*nsx/2.0

                if (numpy.sign(src_affine[1][my])<0):
                    new_affine[1][3] = rsy*nsy/2.0
                else:
                    new_affine[1][3] = -rsy*nsy/2.0

                if (numpy.sign(src_affine[2][mz])<0):
                    new_affine[2][3] = rsz*nsz/2.0
                else:
                    new_affine[2][3] = -rsz*nsz/2.0
            new_affine[3][3] = 1.0

            src_img = nibabel.Nifti1Image(source.get_fdata(), new_affine, source.header)
            src_img.update_header()
            src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcimg'+str(idx)))
            save_volume(src_img_file, src_img)
            source = load_volume(src_img_file)
            src_affine = source.affine
            src_header = source.header
            src_img_files.append(src_img_file)

            # create generic affine aligned with the orientation for the target
            mx = numpy.argmax(numpy.abs(trg_affine[0][0:3]))
            my = numpy.argmax(numpy.abs(trg_affine[1][0:3]))
            mz = numpy.argmax(numpy.abs(trg_affine[2][0:3]))
            new_affine = numpy.zeros((4,4))
            if ignore_res:
                new_affine[0][:] = trg_affine[0][:]/rtx
                new_affine[1][:] = trg_affine[1][:]/rty
                new_affine[2][:] = trg_affine[2][:]/rtz
                rtx = 1.0
                rty = 1.0
                rtz = 1.0

            if ignore_orient:
                new_affine[0][0] = rtx
                new_affine[1][1] = rty
                new_affine[2][2] = rtz
                new_affine[0][3] = -rtx*ntx/2.0
                new_affine[1][3] = -rty*nty/2.0
                new_affine[2][3] = -rtz*ntz/2.0
            elif ignore_affine:
                new_affine[0][mx] = rtx*numpy.sign(trg_affine[0][mx])
                new_affine[1][my] = rty*numpy.sign(trg_affine[1][my])
                new_affine[2][mz] = rtz*numpy.sign(trg_affine[2][mz])
                if (numpy.sign(trg_affine[0][mx])<0):
                    new_affine[0][3] = rtx*ntx/2.0
                else:
                    new_affine[0][3] = -rtx*ntx/2.0

                if (numpy.sign(trg_affine[1][my])<0):
                    new_affine[1][3] = rty*nty/2.0
                else:
                    new_affine[1][3] = -rty*nty/2.0

                if (numpy.sign(trg_affine[2][mz])<0):
                    new_affine[2][3] = rtz*ntz/2.0
                else:
                    new_affine[2][3] = -rtz*ntz/2.0
            new_affine[3][3] = 1.0

            trg_img = nibabel.Nifti1Image(target.get_fdata(), new_affine, target.header)
            trg_img.update_header()
            trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgimg'+str(idx)))
            save_volume(trg_img_file, trg_img)
            target = load_volume(trg_img_file)
            trg_affine = target.affine
            trg_header = target.header
            trg_img_files.append(trg_img_file)

        sources.append(source)
        targets.append(target)

    # build coordinate mapping matrices and save them to disk
    src_coordX = numpy.zeros((nsx,nsy))
    src_coordY = numpy.zeros((nsx,nsy))
    trg_coordX = numpy.zeros((ntx,nty))
    trg_coordY = numpy.zeros((ntx,nty))
    for x in range(nsx):
        for y in range(nsy):
            src_coordX[x,y] = x
            src_coordY[x,y] = y
    src_mapX = nibabel.Nifti1Image(src_coordX, source.affine, source.header)
    src_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordX'))
    save_volume(src_mapX_file, src_mapX)
    src_mapY = nibabel.Nifti1Image(src_coordY, source.affine, source.header)
    src_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordY'))
    save_volume(src_mapY_file, src_mapY)

    for x in range(ntx):
        for y in range(nty):
            trg_coordX[x,y] = x
            trg_coordY[x,y] = y
    trg_mapX = nibabel.Nifti1Image(trg_coordX, target.affine, target.header)
    trg_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordX'))
    save_volume(trg_mapX_file, trg_mapX)
    trg_mapY = nibabel.Nifti1Image(trg_coordY, target.affine, target.header)
    trg_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordY'))
    save_volume(trg_mapY_file, trg_mapY)

    if mask_zero:
        # create and save temporary masks
        target = targets[0]
        trg_mask_data = (target.get_fdata()!=0)
        trg_mask = nibabel.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)

        source = sources[0]
        src_mask_data = (source.get_fdata()!=0)
        src_mask = nibabel.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # run the main ANTS software: here we directly build the command line call
    args = ['--collapse-output-transforms','1',
            '--dimensionality','2',
            '--initialize-transforms-per-stage','0',
    '--interpolation','Linear']

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(module=__name__,file_name=file_name,
                            rootfile=source_images[0],
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    args.append('--output')
    args.append(prefix)

    if mask_zero:
        args.append('--masks')
        args.append('['+trg_mask_file+', '+src_mask_file+']')

    srcfiles = []
    trgfiles = []
    for idx,img in enumerate(sources):
        print("registering "+sources[idx].get_filename()+"\n to "+targets[idx].get_filename())
        srcfiles.append(sources[idx].get_filename())
        trgfiles.append(targets[idx].get_filename())

    weights = []    
    if image_weights is not None:
        weight_sum = 0.0
        for idx,img in enumerate(sources):
            weight_sum = weight_sum + image_weights[idx]
        for idx,img in enumerate(sources):
            weights.append(image_weights[idx]/weight_sum)
    else:        
        for idx,img in enumerate(sources):
            weights.append(1.0/len(srcfiles))

    # figure out the number of scales, going with a factor of two
    n_scales = math.ceil(math.log(scaling_factor)/math.log(2.0))
    iter_rigid = str(rigid_iterations)
    iter_similarity = str(similarity_iterations)
    iter_affine = str(affine_iterations)
    iter_syn = str(coarse_iterations)
    smooth = str(float(scaling_factor))
    shrink = str(scaling_factor)
    for n in range(n_scales):
        iter_rigid = iter_rigid+'x'+str(rigid_iterations)
        iter_similarity = iter_similarity+'x'+str(similarity_iterations)
        iter_affine = iter_affine+'x'+str(affine_iterations)
        if n<(n_scales-1)/2: iter_syn = iter_syn+'x'+str(coarse_iterations)
        elif n<n_scales-1: iter_syn = iter_syn+'x'+str(medium_iterations)
        else: iter_syn = iter_syn+'x'+str(fine_iterations)
        smooth = smooth+'x'+str(scaling_factor/math.pow(2.0,n+1))
        shrink = shrink+'x'+str(math.ceil(scaling_factor/math.pow(2.0,n+1)))

    # set parameters for all the different types of transformations
    if run_rigid is True:
        args.append('--transform')
        args.append('Rigid[0.1]')
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',32,Random,0.3]')

        args.append('--convergence') 
        args.append('['+iter_rigid+','+str(convergence)+',10]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[ 0.001, 0.999 ]')

    if run_similarity is True:
        args.append('--transform')
        args.append('Similarity[0.1]')
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',32,Random,0.3]')

        args.append('--convergence') 
        args.append('['+iter_similarity+','+str(convergence)+',10]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[ 0.001, 0.999 ]')

    if run_affine is True:
        args.append('--transform')
        args.append('Affine[0.1]')
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',32,Random,0.3]')

        args.append('--convergence')
        args.append('['+iter_affine+','+str(convergence)+',10]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[0.001,0.999]')

    if run_syn is True:
        if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
        else: syn_param = [0.2, 3.0, 0.0]

        args.append('--transform')
        args.append('SyN'+str(syn_param))
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',32,Random,0.3]')

        args.append('--convergence')
        args.append('['+iter_syn+','+str(convergence)+',5]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[0.001,0.999]')

    if run_rigid is False and run_similarity is False and run_affine is False and run_syn is False:
        args.append('--transform')
        args.append('Rigid[0.1]')
        for idx,img in enumerate(srcfiles):
            args.append('--metric')
            args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weights[idx])+',5,Random,0.3]')
        args.append(' --convergence')
        args.append('[0,1.0,2]')
        
        args.append('--smoothing-sigmas')
        args.append('0.0')
        args.append('--shrink-factors')
        args.append('1')
        args.append('--use-histogram-matching')
        args.append('0')
        args.append('--winsorize-image-intensities')
        args.append('[0.001,0.999]')

    args.append('--write-composite-transform')
    args.append('0')

    # run the ANTs command directly
    processed_args = ants.utils._int_antsProcessArguments(args)
    print(processed_args)
    libfn = ants.utils.get_lib_fn("antsRegistration")
    libfn(processed_args)

    # output file names
    results = sorted(glob(prefix+'*'))
    forward = []
    flag = []
    for res in results:
        if res.endswith('GenericAffine.mat'):
            forward.append(res)
            flag.append(False)
        elif res.endswith('Warp.nii.gz') and not res.endswith('InverseWarp.nii.gz'):
            forward.append(res)
            flag.append(False)

    #print('forward transforms: '+str(forward))

    inverse = []
    linear = []
    for res in results[::-1]:
        if res.endswith('GenericAffine.mat'):
            inverse.append(res)
            linear.append(True)
        elif res.endswith('InverseWarp.nii.gz'):
            inverse.append(res)
            linear.append(False)

    #print('inverse transforms: '+str(inverse))

    # Transforms the moving image
    for idx,source in enumerate(sources):
        at = ['--dimensionality','2','--input-image-type','0']
        at.append('--input')
        at.append(sources[idx].get_filename())
        at.append('--reference-image')
        at.append(targets[idx].get_filename())
        at.append('--interpolation')
        at.append(interpolation)
        for idx2,transform in enumerate(forward):
            if flag[idx2]:
                at.append('--transform')
                at.append('['+transform+',1]')
            else:
                at.append('--transform')
                at.append('['+transform+',0]')
        at.append('--output')
        at.append(transformed_source_files[idx])

        processed_at = ants.utils._int_antsProcessArguments(at)
        print(processed_at)
        libfn = ants.utils.get_lib_fn("antsApplyTransforms")
        libfn(processed_at)

    # Create coordinate mappings
    src_at = ['--dimensionality','2','--input-image-type','0']
    src_at.append('--input')
    src_at.append(src_mapX.get_filename())
    src_at.append('--reference-image')
    src_at.append(target.get_filename())
    src_at.append('--interpolation')
    src_at.append('Linear')
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at.append('--transform')
            src_at.append('['+transform+',1]')
        else:
            src_at.append('--transform')
            src_at.append('['+transform+',0]')
    src_at.append('--output')
    src_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX_map'))
    src_at.append(src_mapX_trans)

    processed_src_at = ants.utils._int_antsProcessArguments(src_at)
    print(processed_src_at)
    libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    libfn(processed_src_at)

    src_at = ['--dimensionality','2','--input-image-type','0']
    src_at.append('--input')
    src_at.append(src_mapY.get_filename())
    src_at.append('--reference-image')
    src_at.append(target.get_filename())
    src_at.append('--interpolation')
    src_at.append('Linear')
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at.append('--transform')
            src_at.append('['+transform+',1]')
        else:
            src_at.append('--transform')
            src_at.append('['+transform+',0]')
    src_at.append('--output')
    src_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY_map'))
    src_at.append(src_mapY_trans)

    processed_src_at = ants.utils._int_antsProcessArguments(src_at)
    print(processed_src_at)
    libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    libfn(processed_src_at)

    # combine X,Y mappings
    mapX = load_volume(src_mapX_trans).get_fdata()
    mapY = load_volume(src_mapY_trans).get_fdata()
    src_map = numpy.stack((mapX,mapY),axis=-1)
    mapping = nibabel.Nifti1Image(src_map, target.affine, target.header)
    save_volume(mapping_file, mapping)


    trans_mapping = []

    trg_at = ['--dimensionality','2','--input-image-type','0']
    trg_at.append('--input')
    trg_at.append(trg_mapX.get_filename())
    trg_at.append('--reference-image')
    trg_at.append(source.get_filename())
    trg_at.append('--interpolation')
    trg_at.append('Linear')
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at.append('--transform')
            trg_at.append('['+transform+',1]')
        else:
            trg_at.append('--transform')
            trg_at.append('['+transform+',0]')
    trg_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordX_map'))
    trg_at.append('--output')
    trg_at.append(trg_mapX_trans)

    processed_trg_at = ants.utils._int_antsProcessArguments(trg_at)
    print(processed_trg_at)
    libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    libfn(processed_trg_at)
    
    trg_at = ['--dimensionality','2','--input-image-type','0']
    trg_at.append('--input')
    trg_at.append(trg_mapY.get_filename())
    trg_at.append('--reference-image')
    trg_at.append(source.get_filename())
    trg_at.append('--interpolation')
    trg_at.append('Linear')
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at.append('--transform')
            trg_at.append('['+transform+',1]')
        else:
            trg_at.append('--transform')
            trg_at.append('['+transform+',0]')
    trg_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordY_map'))
    trg_at.append('--output')
    trg_at.append(trg_mapY_trans)

    processed_trg_at = ants.utils._int_antsProcessArguments(trg_at)
    print(processed_trg_at)
    libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    libfn(processed_trg_at)
    
    # combine X,Y mappings
    mapX = load_volume(trg_mapX_trans).get_fdata()
    mapY = load_volume(trg_mapY_trans).get_fdata()
    trg_map = numpy.stack((mapX,mapY),axis=-1)
    inverse_mapping = nibabel.Nifti1Image(trg_map, source.affine, source.header)
    save_volume(inverse_mapping_file, inverse_mapping)

    # pad coordinate mapping outside the image? hopefully not needed...

    # clean-up intermediate files
    if os.path.exists(src_mapX_file): os.remove(src_mapX_file)
    if os.path.exists(src_mapY_file): os.remove(src_mapY_file)
    if os.path.exists(trg_mapX_file): os.remove(trg_mapX_file)
    if os.path.exists(trg_mapY_file): os.remove(trg_mapY_file)
    if os.path.exists(src_mapX_trans): os.remove(src_mapX_trans)
    if os.path.exists(src_mapY_trans): os.remove(src_mapY_trans)
    if os.path.exists(trg_mapX_trans): os.remove(trg_mapX_trans)
    if os.path.exists(trg_mapY_trans): os.remove(trg_mapY_trans)
    if ignore_affine or ignore_orient or ignore_res:
        for src_img_file in src_img_files:
            if os.path.exists(src_img_file): os.remove(src_img_file)
        for trg_img_file in trg_img_files:
            if os.path.exists(trg_img_file): os.remove(trg_img_file)

    for name in forward:
        if os.path.exists(name): os.remove(name)
    for name in inverse:
        if os.path.exists(name): os.remove(name)

    # if ignoring header and/or affine, must paste back the correct headers
    if ignore_affine or ignore_orient or ignore_res:
        mapping = load_volume(mapping_file)
        save_volume(mapping_file, nibabel.Nifti1Image(mapping.get_fdata(), orig_trg_aff, orig_trg_hdr))
        inverse = load_volume(inverse_mapping_file)
        save_volume(inverse_mapping_file, nibabel.Nifti1Image(inverse.get_fdata(), orig_src_aff, orig_src_hdr))
        for trans_file in transformed_source_files:
            trans = load_volume(trans_file)
            save_volume(trans_file, nibabel.Nifti1Image(trans.get_fdata(), orig_trg_aff, orig_trg_hdr))

    if not save_data:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(load_volume(trans_file))
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': load_volume(mapping_file),
              'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        for idx,trans_image in enumerate(transformed_source_files):
            if os.path.exists(trans_image): os.remove(trans_image)
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(trans_file)
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': mapping_file,
              'inverse': inverse_mapping_file}

        return output

def embedded_antspy2_multi(source_images, target_images,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False, smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTSpy Registration Multi-contrasts

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines. Uses all input contrasts
    with equal weights.

    Parameters
    ----------
    source_images: [niimg]
        Image list to register
    target_images: [niimg]
        Reference image list to match
    run_rigid: bool
        Whether or not to run a rigid registration first (default is False)
    rigid_iterations: float
        Number of iterations in the rigid step (default is 1000)
    run_similarity: bool
        Whether or not to run a similarity (rigid+scale) registration first (default is False)
    similarity_iterations: float
        Number of iterations in the similarity step (default is 1000)
    run_affine: bool
        Whether or not to run a affine registration first (default is False)
    affine_iterations: float
        Number of iterations in the affine step (default is 1000)
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
    regularization: {'Low', 'Medium', 'High'}
        Regularization preset for the SyN deformation (default is 'Medium')
    convergence: float
        Threshold for convergence, can make the algorithm very slow (default is convergence)
    mask_zero: bool
        Mask regions with zero value using ANTs masking option (default is False)
    smooth_mask: float
        Smoothly mask regions within a given ratio of the object's thickness,
        in [0.0, 1.0] (default is 0.0). This does not use ANTs masking.
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_header: bool
        Ignore the orientation information and affine matrix information
        extracted from the image header (default is False)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * transformed_sources ([niimg]): Deformed source image list (_ants_def0,1,...)
        * mapping (niimg): Coordinate mapping from source to target (_ants_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_ants_invmap)

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. Parameters
    have been set to values commonly found in neuroimaging scripts online, but
    not necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    """

    print('\nEmbedded ANTs Registration Multi-contrasts')

    # make sure that saving related parameters are correct

     # output files needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_images[0])

    transformed_source_files = []
    for idx,source_image in enumerate(source_images):
        transformed_source_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-def'+str(idx))))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-invmap'))
    if save_data:
        if overwrite is False \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            missing = False
            for trans_file in transformed_source_files:
                if not os.path.isfile(trans_file):
                    missing = True

            if not missing:
                print("skip computation (use existing results)")
                transformed = []
                for trans_file in transformed_source_files:
                    transformed.append(trans_file)
                output = {'transformed_sources': transformed,
                      'transformed_source': transformed[0],
                      'mapping': mapping_file,
                      'inverse': inverse_mapping_file}
                return output

    # load and get dimensions and resolution from input images
    sources = []
    targets = []
    for idx,img in enumerate(source_images):
        source = load_volume(source_images[idx])
        src_affine = source.affine
        src_header = source.header
        nsx = source.header.get_data_shape()[X]
        nsy = source.header.get_data_shape()[Y]
        nsz = source.header.get_data_shape()[Z]
        rsx = source.header.get_zooms()[X]
        rsy = source.header.get_zooms()[Y]
        rsz = source.header.get_zooms()[Z]

        orig_src_aff = source.affine
        orig_src_hdr = source.header

        target = load_volume(target_images[idx])
        trg_affine = target.affine
        trg_header = target.header
        ntx = target.header.get_data_shape()[X]
        nty = target.header.get_data_shape()[Y]
        ntz = target.header.get_data_shape()[Z]
        rtx = target.header.get_zooms()[X]
        rty = target.header.get_zooms()[Y]
        rtz = target.header.get_zooms()[Z]

        orig_trg_aff = target.affine
        orig_trg_hdr = target.header

        # in case the affine transformations are not to be trusted: make them equal
        if ignore_affine or ignore_header:
            # create generic affine aligned with the orientation for the source
            new_affine = numpy.zeros((4,4))
            if ignore_header:
                new_affine[0][0] = rsx
                new_affine[1][1] = rsy
                new_affine[2][2] = rsz
                new_affine[0][3] = -rsx*nsx/2.0
                new_affine[1][3] = -rsy*nsy/2.0
                new_affine[2][3] = -rsz*nsz/2.0
            else:
                mx = numpy.argmax(numpy.abs([src_affine[0][0],src_affine[1][0],src_affine[2][0]]))
                my = numpy.argmax(numpy.abs([src_affine[0][1],src_affine[1][1],src_affine[2][1]]))
                mz = numpy.argmax(numpy.abs([src_affine[0][2],src_affine[1][2],src_affine[2][2]]))
                new_affine[mx][0] = rsx*numpy.sign(src_affine[mx][0])
                new_affine[my][1] = rsy*numpy.sign(src_affine[my][1])
                new_affine[mz][2] = rsz*numpy.sign(src_affine[mz][2])
                if (numpy.sign(src_affine[mx][0])<0):
                    new_affine[mx][3] = rsx*nsx/2.0
                else:
                    new_affine[mx][3] = -rsx*nsx/2.0

                if (numpy.sign(src_affine[my][1])<0):
                    new_affine[my][3] = rsy*nsy/2.0
                else:
                    new_affine[my][3] = -rsy*nsy/2.0

                if (numpy.sign(src_affine[mz][2])<0):
                    new_affine[mz][3] = rsz*nsz/2.0
                else:
                    new_affine[mz][3] = -rsz*nsz/2.0
            new_affine[3][3] = 1.0

            src_img = nibabel.Nifti1Image(source.get_fdata(), new_affine, source.header)
            src_img.update_header()
            src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcimg'+str(idx)))
            save_volume(src_img_file, src_img)
            source = load_volume(src_img_file)
            src_affine = source.affine
            src_header = source.header

            # create generic affine aligned with the orientation for the target
            new_affine = numpy.zeros((4,4))
            if ignore_header:
                new_affine[0][0] = rtx
                new_affine[1][1] = rty
                new_affine[2][2] = rtz
                new_affine[0][3] = -rtx*ntx/2.0
                new_affine[1][3] = -rty*nty/2.0
                new_affine[2][3] = -rtz*ntz/2.0
            else:
                mx = numpy.argmax(numpy.abs([trg_affine[0][0],trg_affine[1][0],trg_affine[2][0]]))
                my = numpy.argmax(numpy.abs([trg_affine[0][1],trg_affine[1][1],trg_affine[2][1]]))
                mz = numpy.argmax(numpy.abs([trg_affine[0][2],trg_affine[1][2],trg_affine[2][2]]))
                #print('mx: '+str(mx)+', my: '+str(my)+', mz: '+str(mz))
                #print('rx: '+str(rtx)+', ry: '+str(rty)+', rz: '+str(rtz))
                new_affine[mx][0] = rtx*numpy.sign(trg_affine[mx][0])
                new_affine[my][1] = rty*numpy.sign(trg_affine[my][1])
                new_affine[mz][2] = rtz*numpy.sign(trg_affine[mz][2])
                if (numpy.sign(trg_affine[mx][0])<0):
                    new_affine[mx][3] = rtx*ntx/2.0
                else:
                    new_affine[mx][3] = -rtx*ntx/2.0

                if (numpy.sign(trg_affine[my][1])<0):
                    new_affine[my][3] = rty*nty/2.0
                else:
                    new_affine[my][3] = -rty*nty/2.0

                if (numpy.sign(trg_affine[mz][2])<0):
                    new_affine[mz][3] = rtz*ntz/2.0
                else:
                    new_affine[mz][3] = -rtz*ntz/2.0
            new_affine[3][3] = 1.0
            #print("\nbefore: "+str(trg_affine))
            #print("\nafter: "+str(new_affine))
            trg_img = nibabel.Nifti1Image(target.get_fdata(), new_affine, target.header)
            trg_img.update_header()
            trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgimg'+str(idx)))
            save_volume(trg_img_file, trg_img)
            target = load_volume(trg_img_file)
            trg_affine = target.affine
            trg_header = target.header
            
        sources.append(source)
        targets.append(target)

    # build coordinate mapping matrices and save them to disk
    src_coordX = numpy.zeros((nsx,nsy,nsz))
    src_coordY = numpy.zeros((nsx,nsy,nsz))
    src_coordZ = numpy.zeros((nsx,nsy,nsz))
    trg_coordX = numpy.zeros((ntx,nty,ntz))
    trg_coordY = numpy.zeros((ntx,nty,ntz))
    trg_coordZ = numpy.zeros((ntx,nty,ntz))
    for x in range(nsx):
        for y in range(nsy):
            for z in range(nsz):
                src_coordX[x,y,z] = x
                src_coordY[x,y,z] = y
                src_coordZ[x,y,z] = z
    src_mapX = nibabel.Nifti1Image(src_coordX, source.affine, source.header)
    src_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordX'))
    save_volume(src_mapX_file, src_mapX)
    src_mapY = nibabel.Nifti1Image(src_coordY, source.affine, source.header)
    src_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordY'))
    save_volume(src_mapY_file, src_mapY)
    src_mapZ = nibabel.Nifti1Image(src_coordZ, source.affine, source.header)
    src_mapZ_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordZ'))
    save_volume(src_mapZ_file, src_mapZ)
    for x in range(ntx):
        for y in range(nty):
            for z in range(ntz):
                trg_coordX[x,y,z] = x
                trg_coordY[x,y,z] = y
                trg_coordZ[x,y,z] = z
    trg_mapX = nibabel.Nifti1Image(trg_coordX, target.affine, target.header)
    trg_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordX'))
    save_volume(trg_mapX_file, trg_mapX)
    trg_mapY = nibabel.Nifti1Image(trg_coordY, target.affine, target.header)
    trg_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordY'))
    save_volume(trg_mapY_file, trg_mapY)
    trg_mapZ = nibabel.Nifti1Image(trg_coordZ, target.affine, target.header)
    trg_mapZ_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordZ'))
    save_volume(trg_mapZ_file, trg_mapZ)

    if mask_zero:
        # create and save temporary masks
        target = targets[0]
        trg_mask_data = (target.get_fdata()!=0)
        trg_mask = nibabel.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)

        source = sources[0]
        src_mask_data = (source.get_fdata()!=0)
        src_mask = nibabel.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # if mask boundary regions need to be smoothed away
    if smooth_mask>0:
        # get mask
        mask = nibabel.Nifti1Image(sources[0].get_fdata()>0,sources[0].affine, sources[0].header)
        
        # compute levelset and skeleton
        lvl = probability_to_levelset(mask)['result']
        thk = levelset_thickness(lvl)['dist']
        
        # compute ratio
        ratio = -lvl.get_fdata()/(thk.get_fdata()-lvl.get_fdata())
        ratio[lvl.get_fdata()>0] = 0
        ratio[thk.get_fdata()<=0] = 1
        
        ratio[ratio>smooth_mask] = 1
        ratio[ratio<=smooth_mask] = ratio[ratio<=smooth_mask]/smooth_mask
        
        # multiply all inputs
        for idx,img in enumerate(sources):
            img = nibabel.Nifti1Image(img.get_fdata()*ratio,img.affine, img.header)
            src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcimg'+str(idx)))
            save_volume(src_img_file, img)
            sources[idx] = img

        # get mask
        mask = nibabel.Nifti1Image(targets[0].get_fdata()>0,targets[0].affine, targets[0].header)
        
        # compute levelset and skeleton
        lvl = probability_to_levelset(mask)['result']
        thk = levelset_thickness(lvl)['dist']
        
        # compute ratio
        ratio = -lvl.get_fdata()/(thk.get_fdata()-lvl.get_fdata())
        ratio[lvl.get_fdata()>0] = 0
        ratio[thk.get_fdata()<=0] = 1
        
        ratio[ratio>smooth_mask] = 1
        ratio[ratio<=smooth_mask] = ratio[ratio<=smooth_mask]/smooth_mask
        
        # multiply all inputs
        for idx,img in enumerate(targets):
            img = nibabel.Nifti1Image(img.get_fdata()*ratio,img.affine, img.header)
            trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgimg'+str(idx)))
            save_volume(trg_img_file, img)
            targets[idx] = img


    # run the main ANTS software: here we directly build the command line call
    args = ['--collapse-output-transforms','1',
           '--dimensionality','3',
           '--initialize-transforms-per-stage','0',
           '--interpolation','Linear']

    # add a prefix to avoid multiple names?
    prefix = _fname_4saving(module=__name__,file_name=file_name,
                            rootfile=source_images[0],
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    #reg.inputs.output_transform_prefix = prefix
    args.append('--output')
    args.append(prefix)

    if mask_zero:
        args.append('--masks')
        args.append('['+trg_mask_file+', '+src_mask_file+']')

    srcfiles = []
    trgfiles = []
    for idx,img in enumerate(sources):
        print("registering "+sources[idx].get_filename()+"\n to "+targets[idx].get_filename())
        srcfiles.append(sources[idx].get_filename())
        trgfiles.append(targets[idx].get_filename())

    weight = 1.0/len(srcfiles)

    # figure out the number of scales, going with a factor of two
    n_scales = math.ceil(math.log(scaling_factor)/math.log(2.0))
    iter_rigid = str(rigid_iterations)
    iter_similarity = str(similarity_iterations)
    iter_affine = str(affine_iterations)
    iter_syn = str(coarse_iterations)
    smooth = str(float(scaling_factor))
    shrink = str(scaling_factor)
    for n in range(n_scales):
        iter_rigid = iter_rigid+'x'+str(rigid_iterations)
        iter_similarity = iter_similarity+'x'+str(similarity_iterations)
        iter_affine = iter_affine+'x'+str(affine_iterations)
        if n<(n_scales-1)/2: iter_syn = iter_syn+'x'+str(coarse_iterations)
        elif n<n_scales-1: iter_syn = iter_syn+'x'+str(medium_iterations)
        else: iter_syn = iter_syn+'x'+str(fine_iterations)
        smooth = smooth+'x'+str(scaling_factor/math.pow(2.0,n+1))
        shrink = shrink+'x'+str(math.ceil(scaling_factor/math.pow(2.0,n+1)))

    # set parameters for all the different types of transformations
    if run_rigid is True:
        args.append('--transform')
        args.append('Rigid[0.1]')
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',32,Random,0.3]')

        args.append('--convergence') 
        args.append('['+iter_rigid+','+str(convergence)+',10]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[ 0.001, 0.999 ]')

    if run_similarity is True:
        args.append('--transform')
        args.append('Similarity[0.1]')
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',32,Random,0.3]')

        args.append('--convergence') 
        args.append('['+iter_similarity+','+str(convergence)+',10]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[ 0.001, 0.999 ]')

    if run_affine is True:
        args.append('--transform')
        args.append('Affine[0.1]')
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',32,Random,0.3]')

        args.append('--convergence')
        args.append('['+iter_affine+','+str(convergence)+',10]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[0.001,0.999]')

    if run_syn is True:
        if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
        else: syn_param = [0.2, 3.0, 0.0]

        args.append('--transform')
        args.append('SyN'+str(syn_param))
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',5,Random,0.3]')
        else:
            for idx,img in enumerate(srcfiles):
                args.append('--metric')
                args.append('MI['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',32,Random,0.3]')

        args.append('--convergence')
        args.append('['+iter_syn+','+str(convergence)+',5]')

        args.append('--smoothing-sigmas')
        args.append(smooth)
        
        args.append('--shrink-factors')
        args.append(shrink)
        
        args.append('--use-histogram-matching')
        args.append('0')
        
        args.append('--winsorize-image-intensities')
        args.append('[0.001,0.999]')

    if run_rigid is False and run_similarity is False and run_affine is False and run_syn is False:
        args.append('--transform')
        args.append('Rigid[0.1]')
        for idx,img in enumerate(srcfiles):
            args.append('--metric')
            args.append('CC['+trgfiles[idx]+','+srcfiles[idx] \
                            +','+'{:.3f}'.format(weight)+',5,Random,0.3]')
        args.append(' --convergence')
        args.append('[0,1.0,2]')
        
        args.append('--smoothing-sigmas')
        args.append('0.0')
        args.append('--shrink-factors')
        args.append('1')
        args.append('--use-histogram-matching')
        args.append('0')
        args.append('--winsorize-image-intensities')
        args.append('[0.001,0.999]')

    args.append('--write-composite-transform')
    args.append('0')

    # run the ANTs command directly
    processed_args = ants.utils._int_antsProcessArguments(args)
    print(processed_args)
    libfn = ants.utils.get_lib_fn("antsRegistration")
    libfn(processed_args)
                
    # output file names
    results = sorted(glob(prefix+'*'))
    forward = []
    flag = []
    for res in results:
        if res.endswith('GenericAffine.mat'):
            forward.append(res)
            flag.append(False)
        elif res.endswith('Warp.nii.gz') and not res.endswith('InverseWarp.nii.gz'):
            forward.append(res)
            flag.append(False)

    #print('forward transforms: '+str(forward))

    inverse = []
    linear = []
    for res in results[::-1]:
        if res.endswith('GenericAffine.mat'):
            inverse.append(res)
            linear.append(True)
        elif res.endswith('InverseWarp.nii.gz'):
            inverse.append(res)
            linear.append(False)

    #print('inverse transforms: '+str(inverse))

    # Transforms the moving image
    for idx,source in enumerate(sources):
        at = ['--dimensionality','3','--input-image-type','0']
        at.append('--input')
        at.append(sources[idx].get_filename())
        at.append('--reference-image')
        at.append(targets[idx].get_filename())
        at.append('--interpolation')
        at.append(interpolation)
        for idx2,transform in enumerate(forward):
            if flag[idx2]:
                at.append('--transform')
                at.append('['+transform+',1]')
            else:
                at.append('--transform')
                at.append('['+transform+',0]')
        at.append('--output')
        at.append(transformed_source_files[idx])

        processed_at = ants.utils._int_antsProcessArguments(at)
        print(processed_at)
        libfn = ants.utils.get_lib_fn("antsApplyTransforms")
        libfn(processed_at)

    # Create coordinate mappings
    src_at = ['--dimensionality','3','--input-image-type','0']
    src_at.append('--input')
    src_at.append(src_mapX.get_filename())
    src_at.append('--reference-image')
    src_at.append(target.get_filename())
    src_at.append('--interpolation')
    src_at.append('Linear')
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at.append('--transform')
            src_at.append('['+transform+',1]')
        else:
            src_at.append('--transform')
            src_at.append('['+transform+',0]')
    src_at.append('--output')
    src_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX_map'))
    src_at.append(src_mapX_trans)

    processed_src_at = ants.utils._int_antsProcessArguments(src_at)
    print(processed_src_at)
    #libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    #libfn(processed_src_at)

    src_mapX_trans_img = ants.apply_transforms(fixed=ants.image_read(target.get_filename()),
                                               moving=ants.image_read(src_mapX.get_filename()),
                                               transformlist=forward,
                                               interpolator='linear',imagetype=0,
                                               whichtoinvert=flag, compose=None,
                                               defaultvalue=0, singleprecision=False, verbose=True)
    ants.image_write(src_mapX_trans_img, src_mapX_trans)                                            

    src_at = ['--dimensionality','3','--input-image-type','0']
    src_at.append('--input')
    src_at.append(src_mapY.get_filename())
    src_at.append('--reference-image')
    src_at.append(target.get_filename())
    src_at.append('--interpolation')
    src_at.append('Linear')
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at.append('--transform')
            src_at.append('['+transform+',1]')
        else:
            src_at.append('--transform')
            src_at.append('['+transform+',0]')
    src_at.append('--output')
    src_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY_map'))
    src_at.append(src_mapY_trans)

    processed_src_at = ants.utils._int_antsProcessArguments(src_at)
    print(processed_src_at)
    #libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    #libfn(processed_src_at)

    src_mapY_trans_img = ants.apply_transforms(fixed=ants.image_read(target.get_filename()),
                                               moving=ants.image_read(src_mapY.get_filename()),
                                               transformlist=forward,
                                               interpolator='linear',imagetype=0,
                                               whichtoinvert=flag, compose=None,
                                               defaultvalue=0, singleprecision=False, verbose=True)
    ants.image_write(src_mapY_trans_img, src_mapY_trans)                                            

    src_at = ['--dimensionality','3','--input-image-type','0']
    src_at.append('--input')
    src_at.append(src_mapZ.get_filename())
    src_at.append('--reference-image')
    src_at.append(target.get_filename())
    src_at.append('--interpolation')
    src_at.append('Linear')
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at.append('--transform')
            src_at.append('['+transform+',1]')
        else:
            src_at.append('--transform')
            src_at.append('['+transform+',0]')
    src_at.append('--output')
    src_mapZ_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordZ_map'))
    src_at.append(src_mapZ_trans)

    processed_src_at = ants.utils._int_antsProcessArguments(src_at)
    print(processed_src_at)
    #libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    #libfn(processed_src_at)

    src_mapZ_trans_img = ants.apply_transforms(fixed=ants.image_read(target.get_filename()),
                                               moving=ants.image_read(src_mapZ.get_filename()),
                                               transformlist=forward,
                                               interpolator='linear',imagetype=0,
                                               whichtoinvert=flag, compose=None,
                                               defaultvalue=0, singleprecision=False, verbose=True)
    ants.image_write(src_mapZ_trans_img, src_mapZ_trans)                                            



    # combine X,Y,Z mappings
    mapX = load_volume(src_mapX_trans).get_fdata()
    mapY = load_volume(src_mapY_trans).get_fdata()
    mapZ = load_volume(src_mapZ_trans).get_fdata()
    src_map = numpy.stack((mapX,mapY,mapZ),axis=-1)
    mapping = nibabel.Nifti1Image(src_map, target.affine, target.header)
    save_volume(mapping_file, mapping)

    trg_at = ['--dimensionality','3','--input-image-type','0']
    trg_at.append('--input')
    trg_at.append(trg_mapX.get_filename())
    trg_at.append('--reference-image')
    trg_at.append(source.get_filename())
    trg_at.append('--interpolation')
    trg_at.append('Linear')
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at.append('--transform')
            trg_at.append('['+transform+',1]')
        else:
            trg_at.append('--transform')
            trg_at.append('['+transform+',0]')
    trg_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordX_map'))
    trg_at.append('--output')
    trg_at.append(trg_mapX_trans)

    processed_trg_at = ants.utils._int_antsProcessArguments(trg_at)
    print(processed_trg_at)
    #libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    #libfn(processed_trg_at)

    trg_mapX_trans_img = ants.apply_transforms(fixed=ants.image_read(source.get_filename()),
                                               moving=ants.image_read(trg_mapX.get_filename()),
                                               transformlist=inverse,
                                               interpolator='linear',imagetype=0,
                                               whichtoinvert=linear, compose=None,
                                               defaultvalue=0, singleprecision=False, verbose=True)
    ants.image_write(trg_mapX_trans_img, trg_mapX_trans)                                            

    
    trg_at = ['--dimensionality','3','--input-image-type','0']
    trg_at.append('--input')
    trg_at.append(trg_mapY.get_filename())
    trg_at.append('--reference-image')
    trg_at.append(source.get_filename())
    trg_at.append('--interpolation')
    trg_at.append('Linear')
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at.append('--transform')
            trg_at.append('['+transform+',1]')
        else:
            trg_at.append('--transform')
            trg_at.append('['+transform+',0]')
    trg_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordY_map'))
    trg_at.append('--output')
    trg_at.append(trg_mapY_trans)

    processed_trg_at = ants.utils._int_antsProcessArguments(trg_at)
    print(processed_trg_at)
    #libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    #libfn(processed_trg_at)

    trg_mapY_trans_img = ants.apply_transforms(fixed=ants.image_read(source.get_filename()),
                                               moving=ants.image_read(trg_mapY.get_filename()),
                                               transformlist=inverse,
                                               interpolator='linear',imagetype=0,
                                               whichtoinvert=linear, compose=None,
                                               defaultvalue=0, singleprecision=False, verbose=True)
    ants.image_write(trg_mapY_trans_img, trg_mapY_trans)                                            


    trg_at = ['--dimensionality','3','--input-image-type','0']
    trg_at.append('--input')
    trg_at.append(trg_mapZ.get_filename())
    trg_at.append('--reference-image')
    trg_at.append(source.get_filename())
    trg_at.append('--interpolation')
    trg_at.append('Linear')
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at.append('--transform')
            trg_at.append('['+transform+',1]')
        else:
            trg_at.append('--transform')
            trg_at.append('['+transform+',0]')
    trg_mapZ_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordZ_map'))
    trg_at.append('--output')
    trg_at.append(trg_mapZ_trans)

    processed_trg_at = ants.utils._int_antsProcessArguments(trg_at)
    print(processed_trg_at)
    #libfn = ants.utils.get_lib_fn("antsApplyTransforms")
    #libfn(processed_trg_at)

    trg_mapZ_trans_img = ants.apply_transforms(fixed=ants.image_read(source.get_filename()),
                                               moving=ants.image_read(trg_mapZ.get_filename()),
                                               transformlist=inverse,
                                               interpolator='linear',imagetype=0,
                                               whichtoinvert=linear, compose=None,
                                               defaultvalue=0, singleprecision=False, verbose=True)
    ants.image_write(trg_mapZ_trans_img, trg_mapZ_trans)                                            


    # combine X,Y, Z mappings
    mapX = load_volume(trg_mapX_trans).get_fdata()
    mapY = load_volume(trg_mapY_trans).get_fdata()
    mapZ = load_volume(trg_mapZ_trans).get_fdata()
    trg_map = numpy.stack((mapX,mapY,mapZ),axis=-1)
    inverse_mapping = nibabel.Nifti1Image(trg_map, source.affine, source.header)
    save_volume(inverse_mapping_file, inverse_mapping)

    # pad coordinate mapping outside the image? hopefully not needed...

    # clean-up intermediate files
    if os.path.exists(src_mapX_file): os.remove(src_mapX_file)
    if os.path.exists(src_mapY_file): os.remove(src_mapY_file)
    if os.path.exists(src_mapZ_file): os.remove(src_mapZ_file)
    if os.path.exists(trg_mapX_file): os.remove(trg_mapX_file)
    if os.path.exists(trg_mapY_file): os.remove(trg_mapY_file)
    if os.path.exists(trg_mapZ_file): os.remove(trg_mapZ_file)
    if os.path.exists(src_mapX_trans): os.remove(src_mapX_trans)
    if os.path.exists(src_mapY_trans): os.remove(src_mapY_trans)
    if os.path.exists(src_mapZ_trans): os.remove(src_mapZ_trans)
    if os.path.exists(trg_mapX_trans): os.remove(trg_mapX_trans)
    if os.path.exists(trg_mapY_trans): os.remove(trg_mapY_trans)
    if os.path.exists(trg_mapZ_trans): os.remove(trg_mapZ_trans)
    if ignore_affine or ignore_header:
        if os.path.exists(src_img_file): os.remove(src_img_file)
        if os.path.exists(trg_img_file): os.remove(trg_img_file)

    for name in forward:
        if os.path.exists(name): os.remove(name)
    for name in inverse:
        if os.path.exists(name): os.remove(name)

    # if ignoring header and/or affine, must paste back the correct headers
    if ignore_affine or ignore_header:
        mapping = load_volume(mapping_file)
        save_volume(mapping_file, nibabel.Nifti1Image(mapping.get_fdata(), orig_trg_aff, orig_trg_hdr))
        inverse = load_volume(inverse_mapping_file)
        save_volume(inverse_mapping_file, nibabel.Nifti1Image(inverse.get_fdata(), orig_src_aff, orig_src_hdr))
        for trans_file in transformed_source_files:
            trans = load_volume(trans_file)
            save_volume(trans_file, nibabel.Nifti1Image(trans.get_fdata(), orig_trg_aff, orig_trg_hdr))
    else:
        print("done")

    if not save_data:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(load_volume(trans_file))
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': load_volume(mapping_file),
              'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        for idx,trans_image in enumerate(transformed_source_files):
            if os.path.exists(trans_image): os.remove(trans_image)
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(trans_file)
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': mapping_file,
              'inverse': inverse_mapping_file}

        return output
