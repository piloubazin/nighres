# -*- coding: utf-8 -*
'''
The first basic pipeline for extraction of the WMH from T1/T2 FLAIR.
@author: alaurent
'''
import os
import cbstools
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.base import traits, File
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.interfaces.fsl.preprocess import BET
from nipype.interfaces.fsl.preprocess import FLIRT
from nipype.interfaces.fsl.maths import ApplyMask, Threshold
from nipype.interfaces.fsl.utils import ImageMaths, Reorient2Std, Split
from nipype.interfaces.fsl.base import FSLCommandInputSpec
from nipype.interfaces.fsl import ImageStats
from nipype.interfaces.utility import Function
#import nighres
from nighres.nighres.wrappers import MGDMSegmentation, EnhanceRegionContrast, ProbabilityToLevelset, DefineMultiRegionPriors, RecursiveRidgeDiffusion, LesionExtraction

class AbcImageMathsInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr="%s", mandatory=True, position=1)
    #in_file2 = File(exists=True, argstr="%s", position=3)
    out_file = File(argstr="%s", position=4, genfile=True, hash_files=False)
    op_string = traits.Str(argstr="%s", position=2,
                           desc="string defining the operation, i. e. -add")
    op_value = traits.Float(argstr="%.4f",position=3,des="value to perform operation with")
    suffix = traits.Str(desc="out_file suffix")
    out_data_type = traits.Enum('char', 'short', 'int', 'float', 'double',
                               'input', argstr="-odt %s", position=5,
                               desc=("output datatype, one of (char, short, "
                                     "int, float, double, input)"))

class AbcImageMaths(ImageMaths):
    
    input_spec = AbcImageMathsInputSpec
    #output_spec = ImageMathsOutputSpec
    
def getElementFromList(inlist,idx,slc=None):
    '''
    For selecting a particular element or slice from a list 
    within a nipype connect statement.
    If the slice is longer than the list, this returns the list
    '''
    if not slc:
        outlist=inlist[idx]
    else:
        if slc == -1:
            outlist=inlist[idx:]
        else:
            outlist=inlist[idx:slc]
    return outlist

def createOutputDir(sub,base,name,nodename):
    import os
    return os.path.join(base,name,'_subject_id_'+sub,nodename)

def getFirstElement(inlist):
    '''
    Get the first element from a list
    '''
    return inlist[0]

def Lesion_extractor(name='Lesion_Extractor',
                     wf_name='Test',
                     base_dir='/homes_unix/alaurent/',
                     input_dir=None,
                     subjects=None,
                     main=None,
                     acc=None,
                     atlas='/homes_unix/alaurent/cbstools-public-master/atlases/brain-segmentation-prior3.0/brain-atlas-quant-3.0.8.txt'):
    
    wf = Workflow(wf_name)
    wf.base_dir = base_dir
    
    #file = open(subjects,"r")
    #subjects = file.read().split("\n")
    #file.close()
    
    # Subject List
    subjectList = Node(IdentityInterface(fields=['subject_id'], mandatory_inputs=True), name="subList")
    subjectList.iterables = ('subject_id', [ sub for sub in subjects if sub != '' and sub !='\n' ] )
    
    # T1w and FLAIR
    scanList = Node(DataGrabber(infields=['subject_id'], outfields=['T1', 'FLAIR']), name="scanList")
    scanList.inputs.base_directory = input_dir
    scanList.inputs.ignore_exception = False
    scanList.inputs.raise_on_empty = True
    scanList.inputs.sort_filelist = True
    #scanList.inputs.template = '%s/%s.nii'
    #scanList.inputs.template_args = {'T1': [['subject_id','T1*']], 
    #                                 'FLAIR': [['subject_id','FLAIR*']]}
    scanList.inputs.template = '%s/anat/%s'
    scanList.inputs.template_args = {'T1': [['subject_id','*_T1w.nii.gz']], 
                                     'FLAIR': [['subject_id','*_FLAIR.nii.gz']]}  
    wf.connect(subjectList, "subject_id", scanList, "subject_id")
    
#     # T1w and FLAIR
#     dg = Node(DataGrabber(outfields=['T1', 'FLAIR']), name="T1wFLAIR")
#     dg.inputs.base_directory = "/homes_unix/alaurent/LesionPipeline"
#     dg.inputs.template = "%s/NIFTI/*.nii.gz"
#     dg.inputs.template_args['T1']=[['7']]
#     dg.inputs.template_args['FLAIR']=[['9']]
#     dg.inputs.sort_filelist=True

    # Reorient Volume
    T1Conv = Node(Reorient2Std(), name="ReorientVolume")
    T1Conv.inputs.ignore_exception = False
    T1Conv.inputs.terminal_output = 'none'
    T1Conv.inputs.out_file = "T1_reoriented.nii.gz"
    wf.connect(scanList, "T1", T1Conv, "in_file")
    
    # Reorient Volume (2)
    T2flairConv = Node(Reorient2Std(), name="ReorientVolume2")
    T2flairConv.inputs.ignore_exception = False
    T2flairConv.inputs.terminal_output = 'none'
    T2flairConv.inputs.out_file = "FLAIR_reoriented.nii.gz"
    wf.connect(scanList, "FLAIR", T2flairConv, "in_file")
    
    # N3 Correction
    T1NUC = Node(N4BiasFieldCorrection(), name="N3Correction")
    T1NUC.inputs.dimension = 3
    T1NUC.inputs.environ = {'NSLOTS': '1'}
    T1NUC.inputs.ignore_exception = False
    T1NUC.inputs.num_threads = 1
    T1NUC.inputs.save_bias = False
    T1NUC.inputs.terminal_output = 'none'
    wf.connect(T1Conv, "out_file", T1NUC , "input_image")
        
    # N3 Correction (2)
    T2flairNUC = Node(N4BiasFieldCorrection(), name="N3Correction2")
    T2flairNUC.inputs.dimension = 3
    T2flairNUC.inputs.environ = {'NSLOTS': '1'}
    T2flairNUC.inputs.ignore_exception = False
    T2flairNUC.inputs.num_threads = 1
    T2flairNUC.inputs.save_bias = False
    T2flairNUC.inputs.terminal_output = 'none'
    wf.connect(T2flairConv, "out_file", T2flairNUC, "input_image")
    
    '''
    #####################
    ### PRE-NORMALIZE ###
    #####################
    To make sure there's no outlier values (negative, or really high) to offset the initialization steps
    '''
    
    # Intensity Range Normalization
    getMaxT1NUC = Node(ImageStats(op_string= '-r'), name="getMaxT1NUC")
    wf.connect(T1NUC,'output_image',getMaxT1NUC,'in_file')
    
    T1NUCirn = Node(AbcImageMaths(),name="IntensityNormalization")
    T1NUCirn.inputs.op_string = "-div"
    T1NUCirn.inputs.out_file = "normT1.nii.gz"
    wf.connect(T1NUC,'output_image',T1NUCirn,'in_file')
    wf.connect(getMaxT1NUC,('out_stat',getElementFromList,1),T1NUCirn,"op_value")
    
    # Intensity Range Normalization (2)
    getMaxT2NUC = Node(ImageStats(op_string= '-r'), name="getMaxT2")
    wf.connect(T2flairNUC,'output_image',getMaxT2NUC,'in_file')
    
    T2NUCirn = Node(AbcImageMaths(),name="IntensityNormalization2")
    T2NUCirn.inputs.op_string = "-div"
    T2NUCirn.inputs.out_file = "normT2.nii.gz"
    wf.connect(T2flairNUC,'output_image',T2NUCirn,'in_file')
    wf.connect(getMaxT2NUC,('out_stat',getElementFromList,1),T2NUCirn,"op_value")
    
    '''
    ########################
    #### COREGISTRATION ####
    ########################
    '''
    
    # Optimized Automated Registration
    T2flairCoreg = Node(FLIRT(), name="OptimizedAutomatedRegistration")
    T2flairCoreg.inputs.output_type = 'NIFTI_GZ'
    wf.connect(T2NUCirn, "out_file", T2flairCoreg, "in_file")
    wf.connect(T1NUCirn, "out_file", T2flairCoreg, "reference")
     
    '''    
    #########################
    #### SKULL-STRIPPING ####
    #########################
    '''
    
    # SPECTRE
    T1ss = Node(BET(), name="SPECTRE")
    T1ss.inputs.frac = 0.45 #0.4
    T1ss.inputs.mask = True
    T1ss.inputs.outline = True
    T1ss.inputs.robust = True
    wf.connect(T1NUCirn, "out_file", T1ss, "in_file")
    
    # Image Calculator
    T2ss = Node(ApplyMask(), name="ImageCalculator")
    wf.connect(T1ss,"mask_file",T2ss,"mask_file")
    wf.connect(T2flairCoreg, "out_file", T2ss, "in_file")
    
    '''
    ####################################
    #### 2nd LAYER OF N3 CORRECTION ####
    ####################################
    This time without the skull: there were some significant amounts of inhomogeneities leftover.
    '''
    
    # N3 Correction (3)
    T1ssNUC = Node(N4BiasFieldCorrection(), name="N3Correction3")
    T1ssNUC.inputs.dimension = 3
    T1ssNUC.inputs.environ = {'NSLOTS': '1'}
    T1ssNUC.inputs.ignore_exception = False
    T1ssNUC.inputs.num_threads = 1
    T1ssNUC.inputs.save_bias = False
    T1ssNUC.inputs.terminal_output = 'none'
    wf.connect(T1ss, "out_file", T1ssNUC, "input_image")
    
    # N3 Correction (4)
    T2ssNUC = Node(N4BiasFieldCorrection(), name="N3Correction4")
    T2ssNUC.inputs.dimension = 3
    T2ssNUC.inputs.environ = {'NSLOTS': '1'}
    T2ssNUC.inputs.ignore_exception = False
    T2ssNUC.inputs.num_threads = 1
    T2ssNUC.inputs.save_bias = False
    T2ssNUC.inputs.terminal_output = 'none'
    wf.connect(T2ss, "out_file", T2ssNUC, "input_image")
    
    '''
    ####################################
    ####    NORMALIZE FOR MGDM      ####
    ####################################
    This normalization is a bit aggressive: only useful to have a 
    cropped dynamic range into MGDM, but possibly harmful to further 
    processing, so the unprocessed images are passed to the subsequent steps.
    '''
    
    # Intensity Range Normalization
    getMaxT1ssNUC = Node(ImageStats(op_string= '-r'), name="getMaxT1ssNUC")
    wf.connect(T1ssNUC,'output_image',getMaxT1ssNUC,'in_file')
    
    T1ssNUCirn = Node(AbcImageMaths(),name="IntensityNormalization3")
    T1ssNUCirn.inputs.op_string = "-div"
    T1ssNUCirn.inputs.out_file = "normT1ss.nii.gz"
    wf.connect(T1ssNUC,'output_image',T1ssNUCirn,'in_file')
    wf.connect(getMaxT1ssNUC,('out_stat',getElementFromList,1),T1ssNUCirn,"op_value")
    
    # Intensity Range Normalization (2)
    getMaxT2ssNUC = Node(ImageStats(op_string= '-r'), name="getMaxT2ssNUC")
    wf.connect(T2ssNUC,'output_image',getMaxT2ssNUC,'in_file')
    
    T2ssNUCirn = Node(AbcImageMaths(),name="IntensityNormalization4")
    T2ssNUCirn.inputs.op_string = "-div"
    T2ssNUCirn.inputs.out_file = "normT2ss.nii.gz"
    wf.connect(T2ssNUC,'output_image',T2ssNUCirn,'in_file')
    wf.connect(getMaxT2ssNUC,('out_stat',getElementFromList,1),T2ssNUCirn,"op_value")
    
    '''
    ####################################
    ####      ESTIMATE CSF PV       ####
    ####################################
    Here we try to get a better handle on CSF voxels to help the segmentation step
    '''
    
    # Recursive Ridge Diffusion
    CSF_pv = Node(RecursiveRidgeDiffusion(),name='estimate_CSF_pv')
    CSF_pv.plugin_args = {'sbatch_args':'--mem 6000'}
    CSF_pv.inputs.ridge_intensities = "dark"
    CSF_pv.inputs.ridge_filter = "2D"
    CSF_pv.inputs.orientation = "undefined"
    CSF_pv.inputs.ang_factor = 1.0
    CSF_pv.inputs.min_scale = 0
    CSF_pv.inputs.max_scale = 3
    CSF_pv.inputs.propagation_model = "diffusion"
    CSF_pv.inputs.diffusion_factor = 0.5
    CSF_pv.inputs.similarity_scale = 0.1
    CSF_pv.inputs.neighborhood_size = 4
    CSF_pv.inputs.max_iter = 100
    CSF_pv.inputs.max_diff = 0.001
    CSF_pv.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,CSF_pv.name),CSF_pv,'output_dir')
    wf.connect(T1ssNUCirn,'out_file',CSF_pv,'input_image')
    
    '''
    ####################################
    ####            MGDM            ####
    ####################################
    '''
    
    # Multi-contrast Brain Segmentation
    MGDM = Node(MGDMSegmentation(),name='MGDM')
    MGDM.plugin_args = {'sbatch_args':'--mem 7000'}
    MGDM.inputs.contrast_type1 = "Mprage3T"
    MGDM.inputs.contrast_type2 = "FLAIR3T"
    MGDM.inputs.contrast_type3 = "PVDURA"
    MGDM.inputs.save_data = True
    MGDM.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,MGDM.name),MGDM,'output_dir')
    wf.connect(T1ssNUCirn,'out_file',MGDM,'contrast_image1')
    wf.connect(T2ssNUCirn,'out_file',MGDM,'contrast_image2')
    wf.connect(CSF_pv,'ridge_pv',MGDM,'contrast_image3')
    
    
    # Enhance Region Contrast 
    ERC = Node(EnhanceRegionContrast(),name='ERC')
    ERC.plugin_args = {'sbatch_args':'--mem 7000'}
    ERC.inputs.enhanced_region = "crwm"
    ERC.inputs.contrast_background = "crgm"
    ERC.inputs.partial_voluming_distance = 2.0
    ERC.inputs.save_data = True
    ERC.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,ERC.name),ERC,'output_dir')
    wf.connect(T1ssNUC,'output_image',ERC,'intensity_image')
    wf.connect(MGDM,'segmentation',ERC,'segmentation_image')
    wf.connect(MGDM,'distance',ERC,'levelset_boundary_image')
    
    # Enhance Region Contrast (2)
    ERC2 = Node(EnhanceRegionContrast(),name='ERC2')
    ERC2.plugin_args = {'sbatch_args':'--mem 7000'}
    ERC2.inputs.enhanced_region = "crwm"
    ERC2.inputs.contrast_background = "crgm"
    ERC2.inputs.partial_voluming_distance = 2.0
    ERC2.inputs.save_data = True
    ERC2.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,ERC2.name),ERC2,'output_dir')
    wf.connect(T2ssNUC,'output_image',ERC2,'intensity_image')
    wf.connect(MGDM,'segmentation',ERC2,'segmentation_image')
    wf.connect(MGDM,'distance',ERC2,'levelset_boundary_image')
    
    # Define Multi-Region Priors
    DMRP = Node(DefineMultiRegionPriors(),name='DefineMultRegPriors')
    DMRP.plugin_args = {'sbatch_args':'--mem 6000'}
    #DMRP.inputs.defined_region = "ventricle-horns"
    #DMRP.inputs.definition_method = "closest-distance"
    DMRP.inputs.distance_offset = 3.0
    DMRP.inputs.save_data = True
    DMRP.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,DMRP.name),DMRP,'output_dir')
    wf.connect(MGDM,'segmentation',DMRP,'segmentation_image')
    wf.connect(MGDM,'distance',DMRP,'levelset_boundary_image')
    
    '''
    ###############################################
    ####      REMOVE VENTRICLE POSTERIOR       ####
    ###############################################
    Due to topology constraints, the ventricles are often not fully segmented:
    here add back all ventricle voxels from the posterior probability (without the topology constraints)
    '''
    
    # Posterior label
    PostLabel = Node(Split(),name='PosteriorLabel')
    PostLabel.inputs.dimension = "t"
    wf.connect(MGDM,'labels', PostLabel,'in_file')

    # Posterior proba
    PostProba = Node(Split(),name='PosteriorProba')
    PostProba.inputs.dimension = "t"
    wf.connect(MGDM,'memberships', PostProba,'in_file')
    
    # Threshold binary mask : ventricle label part 1
    VentLabel1 = Node(Threshold(), name="VentricleLabel1")
    VentLabel1.inputs.thresh = 10.5
    VentLabel1.inputs.direction = "below"
    wf.connect(PostLabel,("out_files",getFirstElement), VentLabel1, "in_file")
    
    # Threshold binary mask : ventricle label part 2
    VentLabel2 = Node(Threshold(), name="VentricleLabel2")
    VentLabel2.inputs.thresh = 13.5
    VentLabel2.inputs.direction = "above"
    wf.connect(VentLabel1,"out_file", VentLabel2, "in_file")
    
    # Image calculator : ventricle proba
    VentProba = Node(ImageMaths(), name="VentricleProba")
    VentProba.inputs.op_string = "-mul"
    VentProba.inputs.out_file = "ventproba.nii.gz"
    wf.connect(PostProba,("out_files",getFirstElement),VentProba,"in_file")
    wf.connect(VentLabel2,"out_file", VentProba, "in_file2")
    
    # Image calculator : remove inter ventricles
    RmInterVent = Node(ImageMaths(), name="RemoveInterVent")
    RmInterVent.inputs.op_string = "-sub"
    RmInterVent.inputs.out_file = "rmintervent.nii.gz"
    wf.connect(ERC,"region_pv",RmInterVent,"in_file")
    wf.connect(DMRP, "inter_ventricular_pv", RmInterVent, "in_file2")
    
    # Image calculator : add horns
    AddHorns = Node(ImageMaths(), name="AddHorns")
    AddHorns.inputs.op_string = "-add"
    AddHorns.inputs.out_file = "rmvent.nii.gz"
    wf.connect(RmInterVent,"out_file",AddHorns,"in_file")
    wf.connect(DMRP, "ventricular_horns_pv", AddHorns, "in_file2")  
    
    # Image calculator : remove ventricles
    RmVent = Node(ImageMaths(), name="RemoveVentricles")
    RmVent.inputs.op_string = "-sub"
    RmVent.inputs.out_file = "rmvent.nii.gz"
    wf.connect(AddHorns,"out_file",RmVent,"in_file")
    wf.connect(VentProba, "out_file", RmVent, "in_file2")  
    
    # Image calculator : remove internal capsule
    RmIC = Node(ImageMaths(), name="RemoveInternalCap")
    RmIC.inputs.op_string = "-sub"
    RmIC.inputs.out_file = "rmic.nii.gz"
    wf.connect(RmVent,"out_file",RmIC,"in_file")
    wf.connect(DMRP, "internal_capsule_pv", RmIC, "in_file2")
    
    # Intensity Range Normalization (3)
    getMaxRmIC = Node(ImageStats(op_string= '-r'), name="getMaxRmIC")
    wf.connect(RmIC,'out_file',getMaxRmIC,'in_file')
     
    RmICirn = Node(AbcImageMaths(),name="IntensityNormalization5")
    RmICirn.inputs.op_string = "-div"
    RmICirn.inputs.out_file = "normRmIC.nii.gz"
    wf.connect(RmIC,'out_file',RmICirn,'in_file')
    wf.connect(getMaxRmIC,('out_stat',getElementFromList,1),RmICirn,"op_value")
    
    # Probability To Levelset : WM orientation
    WM_Orient = Node(ProbabilityToLevelset(),name='WM_Orientation')
    WM_Orient.plugin_args = {'sbatch_args':'--mem 6000'}
    WM_Orient.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,WM_Orient.name),WM_Orient,'output_dir')
    wf.connect(RmICirn,'out_file',WM_Orient,'probability_image')

    # Recursive Ridge Diffusion : PVS in WM only
    WM_pvs = Node(RecursiveRidgeDiffusion(),name='PVS_in_WM')
    WM_pvs.plugin_args = {'sbatch_args':'--mem 6000'}
    WM_pvs.inputs.ridge_intensities = "bright"
    WM_pvs.inputs.ridge_filter = "1D"
    WM_pvs.inputs.orientation = "orthogonal"
    WM_pvs.inputs.ang_factor = 1.0
    WM_pvs.inputs.min_scale = 0
    WM_pvs.inputs.max_scale = 3
    WM_pvs.inputs.propagation_model = "diffusion"
    WM_pvs.inputs.diffusion_factor = 1.0
    WM_pvs.inputs.similarity_scale = 1.0
    WM_pvs.inputs.neighborhood_size = 2
    WM_pvs.inputs.max_iter = 100
    WM_pvs.inputs.max_diff = 0.001
    WM_pvs.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,WM_pvs.name),WM_pvs,'output_dir')
    wf.connect(ERC,'background_proba',WM_pvs,'input_image')
    wf.connect(WM_Orient,'levelset',WM_pvs,'surface_levelset')
    wf.connect(RmICirn,'out_file',WM_pvs,'loc_prior')

    # Extract Lesions : extract WM PVS
    extract_WM_pvs = Node(LesionExtraction(),name='ExtractPVSfromWM')
    extract_WM_pvs.plugin_args = {'sbatch_args':'--mem 6000'}
    extract_WM_pvs.inputs.gm_boundary_partial_vol_dist = 1.0
    extract_WM_pvs.inputs.csf_boundary_partial_vol_dist = 3.0
    extract_WM_pvs.inputs.lesion_clust_dist = 1.0
    extract_WM_pvs.inputs.prob_min_thresh = 0.1
    extract_WM_pvs.inputs.prob_max_thresh = 0.33
    extract_WM_pvs.inputs.small_lesion_size = 4.0
    extract_WM_pvs.inputs.save_data = True
    extract_WM_pvs.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_WM_pvs.name),extract_WM_pvs,'output_dir')
    wf.connect(WM_pvs,'propagation',extract_WM_pvs,'probability_image')
    wf.connect(MGDM,'segmentation',extract_WM_pvs,'segmentation_image')
    wf.connect(MGDM,'distance',extract_WM_pvs,'levelset_boundary_image')
    wf.connect(RmICirn,'out_file',extract_WM_pvs,'location_prior_image')
    
    '''
    2nd branch
    '''
    
    # Image calculator : internal capsule witout ventricules
    ICwoVent = Node(ImageMaths(), name="ICWithoutVentricules")
    ICwoVent.inputs.op_string = "-sub"
    ICwoVent.inputs.out_file = "icwovent.nii.gz"
    wf.connect(DMRP,"internal_capsule_pv",ICwoVent,"in_file")
    wf.connect(DMRP,"inter_ventricular_pv", ICwoVent, "in_file2")
    
    # Image calculator : remove ventricles IC
    RmVentIC = Node(ImageMaths(), name="RmVentIC")
    RmVentIC.inputs.op_string = "-sub"
    RmVentIC.inputs.out_file = "RmVentIC.nii.gz"
    wf.connect(ICwoVent,"out_file",RmVentIC,"in_file")
    wf.connect(VentProba, "out_file", RmVentIC, "in_file2")

    # Intensity Range Normalization (4)
    getMaxRmVentIC = Node(ImageStats(op_string= '-r'), name="getMaxRmVentIC")
    wf.connect(RmVentIC,'out_file',getMaxRmVentIC,'in_file')
     
    RmVentICirn = Node(AbcImageMaths(),name="IntensityNormalization6")
    RmVentICirn.inputs.op_string = "-div"
    RmVentICirn.inputs.out_file = "normRmVentIC.nii.gz"
    wf.connect(RmVentIC,'out_file',RmVentICirn,'in_file')
    wf.connect(getMaxRmVentIC,('out_stat',getElementFromList,1),RmVentICirn,"op_value")
    
    # Probability To Levelset : IC orientation
    IC_Orient = Node(ProbabilityToLevelset(),name='IC_Orientation')
    IC_Orient.plugin_args = {'sbatch_args':'--mem 6000'}
    IC_Orient.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,IC_Orient.name),IC_Orient,'output_dir')
    wf.connect(RmVentICirn,'out_file',IC_Orient,'probability_image')
    
    # Recursive Ridge Diffusion : PVS in IC only
    IC_pvs = Node(RecursiveRidgeDiffusion(),name='RecursiveRidgeDiffusion2')
    IC_pvs.plugin_args = {'sbatch_args':'--mem 6000'}
    IC_pvs.inputs.ridge_intensities = "bright"
    IC_pvs.inputs.ridge_filter = "1D"
    IC_pvs.inputs.orientation = "undefined"
    IC_pvs.inputs.ang_factor = 1.0
    IC_pvs.inputs.min_scale = 0
    IC_pvs.inputs.max_scale = 3
    IC_pvs.inputs.propagation_model = "diffusion"
    IC_pvs.inputs.diffusion_factor = 1.0
    IC_pvs.inputs.similarity_scale = 1.0
    IC_pvs.inputs.neighborhood_size = 2
    IC_pvs.inputs.max_iter = 100
    IC_pvs.inputs.max_diff = 0.001
    IC_pvs.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,IC_pvs.name),IC_pvs,'output_dir')
    wf.connect(ERC,'background_proba',IC_pvs,'input_image')
    wf.connect(IC_Orient,'levelset',IC_pvs,'surface_levelset')
    wf.connect(RmVentICirn,'out_file',IC_pvs,'loc_prior')  

    # Extract Lesions : extract IC PVS
    extract_IC_pvs = Node(LesionExtraction(),name='ExtractPVSfromIC')
    extract_IC_pvs.plugin_args = {'sbatch_args':'--mem 6000'}
    extract_IC_pvs.inputs.gm_boundary_partial_vol_dist = 1.0
    extract_IC_pvs.inputs.csf_boundary_partial_vol_dist = 4.0
    extract_IC_pvs.inputs.lesion_clust_dist = 1.0
    extract_IC_pvs.inputs.prob_min_thresh = 0.25
    extract_IC_pvs.inputs.prob_max_thresh = 0.5
    extract_IC_pvs.inputs.small_lesion_size = 4.0
    extract_IC_pvs.inputs.save_data = True
    extract_IC_pvs.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_IC_pvs.name),extract_IC_pvs,'output_dir')
    wf.connect(IC_pvs,'propagation',extract_IC_pvs,'probability_image')
    wf.connect(MGDM,'segmentation',extract_IC_pvs,'segmentation_image')
    wf.connect(MGDM,'distance',extract_IC_pvs,'levelset_boundary_image')
    wf.connect(RmVentICirn,'out_file',extract_IC_pvs,'location_prior_image') 
   
    '''
    3rd branch
    '''
    
    # Image calculator :
    RmInter = Node(ImageMaths(), name="RemoveInterVentricules")
    RmInter.inputs.op_string = "-sub"
    RmInter.inputs.out_file = "rminter.nii.gz"
    wf.connect(ERC2,'region_pv',RmInter,"in_file")
    wf.connect(DMRP,"inter_ventricular_pv", RmInter, "in_file2")
    
    # Image calculator :
    AddVentHorns = Node(ImageMaths(), name="AddVentHorns")
    AddVentHorns.inputs.op_string = "-add"
    AddVentHorns.inputs.out_file = "rminter.nii.gz"
    wf.connect(RmInter,'out_file',AddVentHorns,"in_file")
    wf.connect(DMRP,"ventricular_horns_pv", AddVentHorns, "in_file2")
    
    # Intensity Range Normalization (5)
    getMaxAddVentHorns = Node(ImageStats(op_string= '-r'), name="getMaxAddVentHorns")
    wf.connect(AddVentHorns,'out_file',getMaxAddVentHorns,'in_file')
     
    AddVentHornsirn = Node(AbcImageMaths(),name="IntensityNormalization7")
    AddVentHornsirn.inputs.op_string = "-div"
    AddVentHornsirn.inputs.out_file = "normAddVentHorns.nii.gz"
    wf.connect(AddVentHorns,'out_file',AddVentHornsirn,'in_file')
    wf.connect(getMaxAddVentHorns,('out_stat',getElementFromList,1),AddVentHornsirn,"op_value")
    
    
    # Extract Lesions : extract White Matter Hyperintensities
    extract_WMH = Node(LesionExtraction(),name='Extract_WMH')
    extract_WMH.plugin_args = {'sbatch_args':'--mem 6000'}
    extract_WMH.inputs.gm_boundary_partial_vol_dist = 1.0
    extract_WMH.inputs.csf_boundary_partial_vol_dist = 2.0
    extract_WMH.inputs.lesion_clust_dist = 1.0
    extract_WMH.inputs.prob_min_thresh = 0.84
    extract_WMH.inputs.prob_max_thresh = 0.84
    extract_WMH.inputs.small_lesion_size = 4.0
    extract_WMH.inputs.save_data = True
    extract_WMH.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_WMH.name),extract_WMH,'output_dir')
    wf.connect(ERC2,'background_proba',extract_WMH,'probability_image')
    wf.connect(MGDM,'segmentation',extract_WMH,'segmentation_image')
    wf.connect(MGDM,'distance',extract_WMH,'levelset_boundary_image')
    wf.connect(AddVentHornsirn,'out_file',extract_WMH,'location_prior_image')
    
    #===========================================================================
    # extract_WMH2 = extract_WMH.clone(name='Extract_WMH2')
    # extract_WMH2.inputs.gm_boundary_partial_vol_dist = 2.0
    # wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_WMH2.name),extract_WMH2,'output_dir')
    # wf.connect(ERC2,'background_proba',extract_WMH2,'probability_image')
    # wf.connect(MGDM,'segmentation',extract_WMH2,'segmentation_image')
    # wf.connect(MGDM,'distance',extract_WMH2,'levelset_boundary_image')
    # wf.connect(AddVentHornsirn,'out_file',extract_WMH2,'location_prior_image')
    # 
    # extract_WMH3 = extract_WMH.clone(name='Extract_WMH3')
    # extract_WMH3.inputs.gm_boundary_partial_vol_dist = 3.0
    # wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_WMH3.name),extract_WMH3,'output_dir')
    # wf.connect(ERC2,'background_proba',extract_WMH3,'probability_image')
    # wf.connect(MGDM,'segmentation',extract_WMH3,'segmentation_image')
    # wf.connect(MGDM,'distance',extract_WMH3,'levelset_boundary_image')
    # wf.connect(AddVentHornsirn,'out_file',extract_WMH3,'location_prior_image')
    #===========================================================================
    
    '''
    ####################################
    ####     FINDING SMALL WMHs     ####
    ####################################
    Small round WMHs near the cortex are often missed by the main algorithm, 
    so we're adding this one that takes care of them.
    '''
         
    # Recursive Ridge Diffusion : round WMH detection
    round_WMH = Node(RecursiveRidgeDiffusion(),name='round_WMH')
    round_WMH.plugin_args = {'sbatch_args':'--mem 6000'}
    round_WMH.inputs.ridge_intensities = "bright"
    round_WMH.inputs.ridge_filter = "0D"
    round_WMH.inputs.orientation = "undefined"
    round_WMH.inputs.ang_factor = 1.0
    round_WMH.inputs.min_scale = 1
    round_WMH.inputs.max_scale = 4
    round_WMH.inputs.propagation_model = "none"
    round_WMH.inputs.diffusion_factor = 1.0
    round_WMH.inputs.similarity_scale = 0.1
    round_WMH.inputs.neighborhood_size = 4
    round_WMH.inputs.max_iter = 100
    round_WMH.inputs.max_diff = 0.001
    round_WMH.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,round_WMH.name),round_WMH,'output_dir')
    wf.connect(ERC2,'background_proba',round_WMH,'input_image')
    wf.connect(AddVentHornsirn,'out_file',round_WMH,'loc_prior')  
     
    # Extract Lesions : extract round WMH
    extract_round_WMH = Node(LesionExtraction(),name='Extract_round_WMH')
    extract_round_WMH.plugin_args = {'sbatch_args':'--mem 6000'}
    extract_round_WMH.inputs.gm_boundary_partial_vol_dist = 1.0
    extract_round_WMH.inputs.csf_boundary_partial_vol_dist = 2.0
    extract_round_WMH.inputs.lesion_clust_dist = 1.0
    extract_round_WMH.inputs.prob_min_thresh = 0.33
    extract_round_WMH.inputs.prob_max_thresh = 0.33
    extract_round_WMH.inputs.small_lesion_size = 6.0
    extract_round_WMH.inputs.save_data = True
    extract_round_WMH.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_round_WMH.name),extract_round_WMH,'output_dir')
    wf.connect(round_WMH,'ridge_pv',extract_round_WMH,'probability_image')
    wf.connect(MGDM,'segmentation',extract_round_WMH,'segmentation_image')
    wf.connect(MGDM,'distance',extract_round_WMH,'levelset_boundary_image')
    wf.connect(AddVentHornsirn,'out_file',extract_round_WMH,'location_prior_image')
    
    #===========================================================================
    # extract_round_WMH2 = extract_round_WMH.clone(name='Extract_round_WMH2')
    # extract_round_WMH2.inputs.gm_boundary_partial_vol_dist = 2.0
    # wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_round_WMH2.name),extract_round_WMH2,'output_dir')
    # wf.connect(round_WMH,'ridge_pv',extract_round_WMH2,'probability_image')
    # wf.connect(MGDM,'segmentation',extract_round_WMH2,'segmentation_image')
    # wf.connect(MGDM,'distance',extract_round_WMH2,'levelset_boundary_image')
    # wf.connect(AddVentHornsirn,'out_file',extract_round_WMH2,'location_prior_image')
    # 
    # extract_round_WMH3 = extract_round_WMH.clone(name='Extract_round_WMH3')
    # extract_round_WMH3.inputs.gm_boundary_partial_vol_dist = 2.0
    # wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,extract_round_WMH3.name),extract_round_WMH3,'output_dir')
    # wf.connect(round_WMH,'ridge_pv',extract_round_WMH3,'probability_image')
    # wf.connect(MGDM,'segmentation',extract_round_WMH3,'segmentation_image')
    # wf.connect(MGDM,'distance',extract_round_WMH3,'levelset_boundary_image')
    # wf.connect(AddVentHornsirn,'out_file',extract_round_WMH3,'location_prior_image')
    #===========================================================================
     
    '''
    ####################################
    ####     COMBINE BOTH TYPES     ####
    ####################################
    Small round WMHs and regular WMH together before thresholding
    +
    PVS from white matter and internal capsule
    '''

    # Image calculator : WM + IC DVRS
    DVRS = Node(ImageMaths(), name="DVRS")
    DVRS.inputs.op_string = "-max"
    DVRS.inputs.out_file = "DVRS_map.nii.gz"
    wf.connect(extract_WM_pvs,'lesion_score',DVRS,"in_file")
    wf.connect(extract_IC_pvs,"lesion_score", DVRS, "in_file2")
    
    # Image calculator : WMH + round
    WMH = Node(ImageMaths(), name="WMH")
    WMH.inputs.op_string = "-max"
    WMH.inputs.out_file = "WMH_map.nii.gz"
    wf.connect(extract_WMH,'lesion_score',WMH,"in_file")
    wf.connect(extract_round_WMH,"lesion_score", WMH, "in_file2")
    
    #===========================================================================
    # WMH2 = Node(ImageMaths(), name="WMH2")
    # WMH2.inputs.op_string = "-max"
    # WMH2.inputs.out_file = "WMH2_map.nii.gz"
    # wf.connect(extract_WMH2,'lesion_score',WMH2,"in_file")
    # wf.connect(extract_round_WMH2,"lesion_score", WMH2, "in_file2")
    # 
    # WMH3 = Node(ImageMaths(), name="WMH3")
    # WMH3.inputs.op_string = "-max"
    # WMH3.inputs.out_file = "WMH3_map.nii.gz"
    # wf.connect(extract_WMH3,'lesion_score',WMH3,"in_file")
    # wf.connect(extract_round_WMH3,"lesion_score", WMH3, "in_file2")
    #===========================================================================
    
    # Image calculator : multiply by boundnary partial volume
    WMH_mul = Node(ImageMaths(), name="WMH_mul")
    WMH_mul.inputs.op_string = "-mul"
    WMH_mul.inputs.out_file = "final_mask.nii.gz"
    wf.connect(WMH,"out_file", WMH_mul,"in_file")
    wf.connect(MGDM,"distance", WMH_mul, "in_file2")
    
    #===========================================================================
    # WMH2_mul = Node(ImageMaths(), name="WMH2_mul")
    # WMH2_mul.inputs.op_string = "-mul"
    # WMH2_mul.inputs.out_file = "final_mask.nii.gz"
    # wf.connect(WMH2,"out_file", WMH2_mul,"in_file")
    # wf.connect(MGDM,"distance", WMH2_mul, "in_file2")
    # 
    # WMH3_mul = Node(ImageMaths(), name="WMH3_mul")
    # WMH3_mul.inputs.op_string = "-mul"
    # WMH3_mul.inputs.out_file = "final_mask.nii.gz"
    # wf.connect(WMH3,"out_file", WMH3_mul,"in_file")
    # wf.connect(MGDM,"distance", WMH3_mul, "in_file2")
    #===========================================================================
    
    '''
    ##########################################
    ####      SEGMENTATION THRESHOLD      ####
    ##########################################
    A threshold of 0.5 is very conservative, because the final lesion score is the product of two probabilities.
    This needs to be optimized to a value between 0.25 and 0.5 to balance false negatives 
    (dominant at 0.5) and false positives (dominant at low values).
    '''
    
    # Threshold binary mask : 
    DVRS_mask = Node(Threshold(), name="DVRS_mask")
    DVRS_mask.inputs.thresh = 0.25
    DVRS_mask.inputs.direction = "below"
    wf.connect(DVRS,"out_file", DVRS_mask, "in_file")
    
    # Threshold binary mask : 025
    WMH1_025 = Node(Threshold(), name="WMH1_025")
    WMH1_025.inputs.thresh = 0.25
    WMH1_025.inputs.direction = "below"
    wf.connect(WMH_mul,"out_file", WMH1_025, "in_file")
    
    #===========================================================================
    # WMH2_025 = Node(Threshold(), name="WMH2_025")
    # WMH2_025.inputs.thresh = 0.25
    # WMH2_025.inputs.direction = "below"
    # wf.connect(WMH2_mul,"out_file", WMH2_025, "in_file")
    # 
    # WMH3_025 = Node(Threshold(), name="WMH3_025")
    # WMH3_025.inputs.thresh = 0.25
    # WMH3_025.inputs.direction = "below"
    # wf.connect(WMH3_mul,"out_file", WMH3_025, "in_file")
    #===========================================================================
    
    # Threshold binary mask : 050
    WMH1_050 = Node(Threshold(), name="WMH1_050")
    WMH1_050.inputs.thresh = 0.50
    WMH1_050.inputs.direction = "below"
    wf.connect(WMH_mul,"out_file", WMH1_050, "in_file")
    
    #===========================================================================
    # WMH2_050 = Node(Threshold(), name="WMH2_050")
    # WMH2_050.inputs.thresh = 0.50
    # WMH2_050.inputs.direction = "below"
    # wf.connect(WMH2_mul,"out_file", WMH2_050, "in_file")
    # 
    # WMH3_050 = Node(Threshold(), name="WMH3_050")
    # WMH3_050.inputs.thresh = 0.50
    # WMH3_050.inputs.direction = "below"
    # wf.connect(WMH3_mul,"out_file", WMH3_050, "in_file")
    #===========================================================================
    
    # Threshold binary mask : 075
    WMH1_075 = Node(Threshold(), name="WMH1_075")
    WMH1_075.inputs.thresh = 0.75
    WMH1_075.inputs.direction = "below"
    wf.connect(WMH_mul,"out_file", WMH1_075, "in_file")
    
    #===========================================================================
    # WMH2_075 = Node(Threshold(), name="WMH2_075")
    # WMH2_075.inputs.thresh = 0.75
    # WMH2_075.inputs.direction = "below"
    # wf.connect(WMH2_mul,"out_file", WMH2_075, "in_file")
    # 
    # WMH3_075 = Node(Threshold(), name="WMH3_075")
    # WMH3_075.inputs.thresh = 0.75
    # WMH3_075.inputs.direction = "below"
    # wf.connect(WMH3_mul,"out_file", WMH3_075, "in_file")
    #===========================================================================
    
    ## Outputs
    
    DVRS_Output=Node(IdentityInterface(fields=['mask','region','lesion_size','lesion_proba','boundary','label','score']), name='DVRS_Output')
    wf.connect(DVRS_mask,'out_file',DVRS_Output,'mask')
    
    
    WMH_output=Node(IdentityInterface(fields=['mask1025','mask1050','mask1075','mask2025','mask2050','mask2075','mask3025','mask3050','mask3075']), name='WMH_output')
    wf.connect(WMH1_025,'out_file',WMH_output,'mask1025')
    #wf.connect(WMH2_025,'out_file',WMH_output,'mask2025')
    #wf.connect(WMH3_025,'out_file',WMH_output,'mask3025')
    wf.connect(WMH1_050,'out_file',WMH_output,'mask1050')
    #wf.connect(WMH2_050,'out_file',WMH_output,'mask2050')
    #wf.connect(WMH3_050,'out_file',WMH_output,'mask3050')
    wf.connect(WMH1_075,'out_file',WMH_output,'mask1075')
    #wf.connect(WMH2_075,'out_file',WMH_output,'mask2070')
    #wf.connect(WMH3_075,'out_file',WMH_output,'mask3075')


    return wf
    
