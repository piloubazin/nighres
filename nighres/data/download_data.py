import os
from urllib.request import urlretrieve
from nighres.global_settings import ATLAS_DIR

def download_7T_TRT(data_dir, overwrite=False, subject_id='sub001_sess1'):
    """
    Downloads the MP2RAGE data from the 7T Test-Retest
    dataset published by Gorgolewski et al (2015) [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded files should be stored. A
        subdirectory called '7T_TRT' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)
    subject_id: 'sub001_sess1', 'sub002_sess1', 'sub003_sess1'}
        Which dataset to download (default is 'sub001_sess1')

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * inv2 : path to second inversion image
        * t1w : path to T1-weighted (uniform) image
        * t1map : path to quantitative T1 image

    Notes
    ----------
    The full dataset is available at http://openscience.cbs.mpg.de/7t_trt/

    References
    ----------
    .. [1] Gorgolewski et al (2015). A high resolution 7-Tesla resting-state
       fMRI test-retest dataset with cognitive and physiological measures.
       DOI: 10.1038/sdata.2014.
    """

    data_dir = os.path.join(data_dir, '7T_TRT')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    nitrc = 'https://www.nitrc.org/frs/download.php/'
    if subject_id == 'sub001_sess1':
        file_sources = [nitrc + x for x in ['10234', '10235', '10236']]
    elif subject_id == 'sub002_sess1':
        file_sources = [nitrc + x for x in ['10852', '10853', '10854']]
    elif subject_id == 'sub003_sess1':
        file_sources = [nitrc + x for x in ['10855', '10856', '10857']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    [subject_id+'_INV2.nii.gz',
                     subject_id+'_T1map.nii.gz',
                     subject_id+'_T1w.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'inv2': file_targets[0],
            't1map': file_targets[1],
            't1w': file_targets[2]}


def download_DTI_2mm(data_dir, overwrite=False):
    """
    Downloads an example DTI data set

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded files should be stored. A
        subdirectory called 'DTI_2mm' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * dti : path to DTI image
        * mask : path to binary brain mask

    """

    data_dir = os.path.join(data_dir, 'DTI_2mm')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    nitrc = 'https://www.nitrc.org/frs/download.php/'

    file_sources = [nitrc + x for x in ['11511', '11512']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['DTI_2mm.nii.gz',
                     'DTI_2mm_brain_mask.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'dti': file_targets[0],
            'mask': file_targets[1]}


def download_DOTS_atlas(data_dir=None, overwrite=False):
    """
    Downloads the statistical atlas presented in [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'DOTS_atlas' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * fiber_p : path to atlas probability image
        * fiber_dir : path to atlas direction image

    References
    ----------
    .. [1] Bazin et al (2011). Direct segmentation of the major white matter
           tracts in diffusion tensor images.
           DOI: 10.1016/j.neuroimage.2011.06.020
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'DOTS_atlas')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    nitrc = 'https://www.nitrc.org/frs/download.php/'

    file_sources = [nitrc + x for x in ['11514', '11513']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['fiber_p.nii.gz',
                     'fiber_dir.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'fiber_p': file_targets[0],
            'fiber_dir': file_targets[1]}

def download_MASSP_atlas(data_dir=None, overwrite=False):
    """
    Downloads the MASSP atlas presented in [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'massp-prior' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * histograms : path to histogram image
        * spatial_probas : path to spatial probability image
        * spatial_labels : path to spatial label image
        * skeleton_probas : path to skeleton probability image
        * skeleton_labels : path to skeleton label image

    References
    ----------
    .. [1] Bazin et al (2020). Multi-contrast Anatomical Subcortical
    Structure Parcellation. Under review.
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'massp-prior')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['22627481','22627484','22627475','22627478','22627472']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['massp_17structures_spatial_label.nii.gz',
                     'massp_17structures_spatial_proba.nii.gz',
                     'massp_17structures_skeleton_label.nii.gz',
                     'massp_17structures_skeleton_proba.nii.gz',
                     'massp_17structures_r1r2sqsm_histograms.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'spatial_labels': file_targets[0],
            'spatial_probas': file_targets[1],
            'skeleton_labels': file_targets[2],
            'skeleton_probas': file_targets[3],
            'histograms': file_targets[4]}

def download_MASSP2p0_atlas(data_dir=None, overwrite=False):
    """
    Downloads the MASSP2.0 atlas presented in [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'massp-prior' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * mp2rageme : path to MP2RAGEME histogram image (R1, R2*, QSM, PD in that order)
        * mp2rageme-fcm : path to MP2RAGEME histogram image (same order, assuming FCM normalization)
        * mpm7T : path to MPM 7T histogram image (R1, R2*, PD, QSM, MT in that order)
        * mpm7T-noqsm : path to MPM 7T histogram image (R1, R2*, PD, MT in that order)
        * mpm7T-nopdqsm : path to MPM 7T histogram image (R1, R2*, MT in that order)
        * mpm3T : path to MPM 3T histogram image (R1, R2*, PD, MT in that order)
        * spatial_probas : path to spatial probability image
        * spatial_labels : path to spatial label image
        * skeleton_probas : path to skeleton probability image
        * skeleton_labels : path to skeleton label image

    References
    ----------
    .. [1] Bazin et al (2025). Automated parcellation and atlasing of the human subcortex with 
        ultra-high resolution quantitative MRI. Imaging Neuroscience, 2025.
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'massp-prior')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['50075790','50075796','50075778','50075787',
                     '50075784','50075781',
                     '50295732','50295735','50295738',
                     '50295741']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['massp_2p0_spatial_label.nii.gz',
                     'massp_2p0_spatial_proba.nii.gz',
                     'massp_2p0_skeleton_label.nii.gz',
                     'massp_2p0_skeleton_proba.nii.gz',
                     'massp_2p0_mp2rageme_r1r2sqsmpd_histograms.nii.gz',
                     'massp_2p0_mp2rageme_r1r2sqsmpd_fcm_histograms.nii.gz',
                     'massp_2p0_mpm7T_r1r2spdqsmmt_histograms.nii.gz',
                     'massp_2p0_mpm7T_r1r2spdmt_histograms.nii.gz',
                     'massp_2p0_mpm7T_r1r2smt_histograms.nii.gz',
                     'massp_2p0_mpm3T_r1r2spdmt_histograms.nii.gz',
                     ]]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'spatial_labels': file_targets[0],
            'spatial_probas': file_targets[1],
            'skeleton_labels': file_targets[2],
            'skeleton_probas': file_targets[3],
            'mp2rageme': file_targets[4],
            'mp2rageme-fcm': file_targets[5],
            'mpm7T': file_targets[6],
            'mpm7T-noqsm': file_targets[7],
            'mpm7T-nopdqsm': file_targets[8],
            'mpm3T': file_targets[9]}

def download_MP2RAGEME_sample(data_dir, overwrite=False):
    """
    Downloads an example data set from a MP2RAGEME acquisition _[1].

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * qr1 : path to quantitative R1 map image
        * qr2s : path to quantitative R2* map image
        * qsm : path to QSM image

    References
    ----------
    .. [1] Caan et al (2018). MP2RAGEME: T1, T2*, and QSM mapping in one
    sequence at 7 tesla. doi:10.1002/hbm.24490
    """

    data_dir = os.path.join(data_dir, 'mp2rageme')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['22678334','22678337','22628750']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['sample-subject_mp2rageme-qr1_brain.nii.gz',
                     'sample-subject_mp2rageme-qr2s_brain.nii.gz',
                     'sample-subject_mp2rageme-qsm_brain.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'qr1': file_targets[0],
            'qr2s': file_targets[1],
            'qsm': file_targets[2]}

def download_MP2RAGEME_testdata(data_dir, overwrite=False):
    """
    Downloads a down-sampled example data set from a MP2RAGEME acquisition _[1]
    for testing purposes.

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * mp2rageme_mag : list of path to first and second inversion magnitudes
        * mp2rageme_phs : list of path to first and second inversion phases
        * mp2rageme_inv2e1: path to second inversion, first echo magnitude
        * slab_inv2e1: path to second inversion, first echo magnitude of slab
        * mp2rageme_qt1 : path to quantitative T1 map image
        * slab_qt1 : path to quantitative T1 map image of slab
        
    References
    ----------
    .. [1] Caan et al (2018). MP2RAGEME: T1, T2*, and QSM mapping in one
    sequence at 7 tesla. doi:10.1002/hbm.24490
    """

    data_dir = os.path.join(data_dir, 'test_data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['41074373','41074418','41074424','41074430','41074436',
                     '41074415','41074421','41074427','41074433','41074439',
                     '41074370','41074412','41074376']]
                   

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['test-subject_mp2rageme-inv1m.nii.gz',
                     'test-subject_mp2rageme-inv2e1m.nii.gz',
                     'test-subject_mp2rageme-inv2e2m.nii.gz',
                     'test-subject_mp2rageme-inv2e3m.nii.gz',
                     'test-subject_mp2rageme-inv2e4m.nii.gz',
                     'test-subject_mp2rageme-inv1p.nii.gz',
                     'test-subject_mp2rageme-inv2e1p.nii.gz',
                     'test-subject_mp2rageme-inv2e2p.nii.gz',
                     'test-subject_mp2rageme-inv2e3p.nii.gz',
                     'test-subject_mp2rageme-inv2e4p.nii.gz',
                     'test-subject_mp2rageme-qt1.nii.gz',
                     'test-subject_slab-inv2e1m.nii.gz',
                     'test-subject_slab-qt1.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'mp2rageme_mag': file_targets[0:5],
            'mp2rageme_phs': file_targets[5:10],
            'mp2rageme_inv2e1': file_targets[1],
            'mp2rageme_qt1': file_targets[10],
            'slab_inv2e1': file_targets[11],
            'slab_qt1': file_targets[12]}

def download_AHEAD_template(data_dir=None, overwrite=False):
    """
    Downloads the AHEAD group template _[1].

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'ahead-template' will be created in this location
        (default is ATLAS_DIR)
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * qr1 : path to quantitative R1 map image
        * qr2s : path to quantitative R2* map image
        * qsm : path to QSM image

    References
    ----------
    .. [1] Alkemade et al (under review). The Amsterdam Ultra-high field adult
       lifespan database (AHEAD): A freely available multimodal 7 Tesla
       submillimeter magnetic resonance imaging database.
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'ahead-template')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['22679537','22679543','22679546']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['ahead_med_qr1.nii.gz',
                     'ahead_med_qr2s.nii.gz',
                     'ahead_med_qsm.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'qr1': file_targets[0],
            'qr2s': file_targets[1],
            'qsm': file_targets[2]}

def download_AHEADmni2009b_template(data_dir=None, overwrite=False):
    """
    Downloads the AHEAD group template in MNI 2009b space _[1].

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'ahead-template' will be created in this location
        (default is ATLAS_DIR)
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * qr1 : path to quantitative R1 map image
        * qr2s : path to quantitative R2* map image
        * qsm : path to QSM image
        * qpd : path to semi-quantitative PD image

    References
    ----------
    .. [1] Alkemade et al (under review). The Amsterdam Ultra-high field adult
       lifespan database (AHEAD): A freely available multimodal 7 Tesla
       submillimeter magnetic resonance imaging database.
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'ahead-template')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

# older version of the template
#    file_sources = [figshare + x for x in
#                    ['22679537','22679543','22679546','']]
    file_sources = [figshare + x for x in
                    ['34892901','34892907','34892883','41498217']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['ahead_mni2009b_med_qr1.nii.gz',
                     'ahead_mni2009b_med_qr2s.nii.gz',
                     'ahead_mni2009b_med_qsm.nii.gz',
                     'ahead_mni2009b_med_qpd.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'qr1': file_targets[0],
            'qr2s': file_targets[1],
            'qsm': file_targets[2],
            'qpd': file_targets[3]}

