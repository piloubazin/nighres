# -*- coding: utf-8 -*
'''
The first basic pipeline for extraction of the WMH from T1/T2 FLAIR.
@author: alaurent
'''
import os
import cbstools
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.interfaces.fsl.preprocess import BET
from nipype.interfaces.fsl.preprocess import FLIRT
from nipype.interfaces.fsl.maths import ApplyMask, Threshold
from nipype.interfaces.fsl.utils import ImageMaths, Reorient2Std
from nipype.interfaces.fsl import ImageStats
from ginnipi.toolbox.flow import getElementFromList
from ginnipi.interfaces.custom import AbcImageMaths
from nipype.interfaces.utility import Function
#import nighres
from nighres.nighres.wrappers import MGDMSegmentation, EnhanceRegionContrast, ProbabilityToLevelset, DefineMultiRegionPriors, RecursiveRidgeDiffusion, LesionExtraction

def createOutputDir(sub,base,name,nodename):
    import os
    return os.path.join(base,name,'_subject_id_'+sub,nodename)

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
    
    file = open(subjects,"r")
    subjects = file.read().split("\n")
    file.close()
    
    # Subject List
    subjectList = Node(IdentityInterface(fields=['subject_id'], mandatory_inputs=True), name="subList")
    subjectList.iterables = ('subject_id', [ sub for sub in subjects if sub != '' and sub !='\n' ] )
    
    # T1w and FLAIR
    scanList = Node(DataGrabber(infields=['subject_id'], outfields=['T1', 'FLAIR']), name="scanList")
    scanList.inputs.base_directory = input_dir
    scanList.inputs.ignore_exception = False
    scanList.inputs.raise_on_empty = True
    scanList.inputs.sort_filelist = True
    scanList.inputs.template = '%s/%s.nii'
    scanList.inputs.template_args = {'T1': [['subject_id','T1*']], 
                                     'FLAIR': [['subject_id','FLAIR*']]}    
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
    
    # Optimized Automated Registration
    T2flairCoreg = Node(FLIRT(), name="OptimizedAutomatedRegistration")
    T2flairCoreg.inputs.output_type = 'NIFTI_GZ'
    wf.connect(T2flairNUC, "output_image", T2flairCoreg, "in_file")
    wf.connect(T1NUC, "output_image", T2flairCoreg, "reference")
        
    
    # SPECTRE
    T1ss = Node(BET(), name="SPECTRE")
    T1ss.inputs.frac = 0.4
    T1ss.inputs.mask = True
    T1ss.inputs.outline = True
    T1ss.inputs.robust = True
    wf.connect(T1NUC, "output_image", T1ss, "in_file")
    
    T1ssBIS = Node(BET(), name="SPECTREBis")
    T1ssBIS.inputs.frac = 0.4 # parametre a gerer
    T1ssBIS.inputs.mask = True
    T1ssBIS.inputs.outline = True
    T1ssBIS.inputs.reduce_bias = True
    wf.connect(T1NUC, "output_image", T1ssBIS, "in_file")
    
    # Image Calculator
    T2ss = Node(ApplyMask(), name="ImageCalculator")
    wf.connect(T1ss,"mask_file",T2ss,"mask_file")
    wf.connect(T2flairCoreg, "out_file", T2ss, "in_file")
    
    T2ssBIS = Node(ApplyMask(), name="ImageCalculatorBIS")
    wf.connect(T1ssBIS,"mask_file",T2ssBIS,"mask_file")
    wf.connect(T2flairCoreg, "out_file", T2ssBIS, "in_file")
    
    # Intensity Range Normalization
    getMaxT1 = Node(ImageStats(op_string= '-r'), name="getMaxT1")
    wf.connect(T1ss,'out_file',getMaxT1,'in_file')
    
    T1irn = Node(AbcImageMaths(),name="IntensityNormalization")
    T1irn.inputs.op_string = "-div"
    T1irn.inputs.out_file = "normT1.nii.gz"
    wf.connect(T1ss,'out_file',T1irn,'in_file')
    wf.connect(getMaxT1,('out_stat',getElementFromList,1),T1irn,"op_value")
    
    # Intensity Range Normalization (2)
    getMaxT2 = Node(ImageStats(op_string= '-r'), name="getMaxT2")
    wf.connect(T2ss,'out_file',getMaxT2,'in_file')
    
    T2irn = Node(AbcImageMaths(),name="IntensityNormalization2")
    T2irn.inputs.op_string = "-div"
    T2irn.inputs.out_file = "normT2.nii.gz"
    wf.connect(T2ss,'out_file',T2irn,'in_file')
    wf.connect(getMaxT2,('out_stat',getElementFromList,1),T2irn,"op_value")
    
    # Multi-contrast Brain Segmentation
    MGDM = Node(MGDMSegmentation(),name='MGDM')
    MGDM.plugin_args = {'sbatch_args':'--mem 7000'}
    MGDM.inputs.contrast_type1 = "Mprage3T"
    MGDM.inputs.contrast_type2 = "FLAIR3T"
    MGDM.inputs.save_data = True
    MGDM.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,MGDM.name),MGDM,'output_dir')
    wf.connect(T1irn,'out_file',MGDM,'contrast_image1')
    wf.connect(T2irn,'out_file',MGDM,'contrast_image2')
    
    # Enhance Region Contrast 
    ERC = Node(EnhanceRegionContrast(),name='ERC')
    ERC.plugin_args = {'sbatch_args':'--mem 7000'}
    ERC.inputs.enhanced_region = "crwm"
    ERC.inputs.contrast_background = "crgm"
    ERC.inputs.partial_voluming_distance = 2.0
    ERC.inputs.save_data = True
    ERC.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,ERC.name),ERC,'output_dir')
    wf.connect(T1irn,'out_file',ERC,'intensity_image')
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
    wf.connect(T2irn,'out_file',ERC2,'intensity_image')
    wf.connect(MGDM,'segmentation',ERC2,'segmentation_image')
    wf.connect(MGDM,'distance',ERC2,'levelset_boundary_image')
    
    # Define Multi-Region Priors
    DMRP = Node(DefineMultiRegionPriors(),name='DefineMultRegPriors')
    DMRP.plugin_args = {'sbatch_args':'--mem 6000'}
    #DMRP.inputs.defined_region = "ventricle-horns"
    DMRP.inputs.definition_method = "closest-distance"
    DMRP.inputs.distance_offset = 1.0
    DMRP.inputs.save_data = True
    DMRP.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,DMRP.name),DMRP,'output_dir')
    wf.connect(MGDM,'segmentation',DMRP,'segmentation_image')
    wf.connect(MGDM,'distance',DMRP,'levelset_boundary_image')
    
    
    # Image calculator : remove inter ventricules
    RmVent = Node(ImageMaths(), name="RemoveVentricules")
    RmVent.inputs.op_string = "-sub"
    RmVent.inputs.out_file = "rmvent.nii.gz"
    wf.connect(ERC,"region_pv",RmVent,"in_file")
    wf.connect(DMRP, "inter_ventricular_pv", RmVent, "in_file2")
    
    # Threshold binary mask : ventricule horns
    VentHorns = Node(Threshold(), name="VentriculeHorns")
    VentHorns.inputs.thresh = 0.5
    VentHorns.inputs.direction = "above"
    wf.connect(DMRP, "ventricular_horns_pv", VentHorns, "in_file")
     
    # Image calculator : add horns
    AddHorns = Node(ImageMaths(), name="AddHorns")
    AddHorns.inputs.op_string = "-add"
    AddHorns.inputs.out_file = "rmvent.nii.gz"
    wf.connect(RmVent,"out_file",AddHorns,"in_file")
    wf.connect(VentHorns, "out_file", AddHorns, "in_file2")
     
    # Image calculator : remove internal capsule
    RmIC = Node(ImageMaths(), name="RemoveInternalCap")
    RmIC.inputs.op_string = "-sub"
    RmIC.inputs.out_file = "rmic.nii.gz"
    wf.connect(AddHorns,"out_file",RmIC,"in_file")
    wf.connect(DMRP, "internal_capsule_pv", RmIC, "in_file2")
     
    # Intensity Range Normalization (3)
    getMaxRmIC = Node(ImageStats(op_string= '-r'), name="getMaxRmIC")
    wf.connect(RmIC,'out_file',getMaxRmIC,'in_file')
     
    RmICirn = Node(AbcImageMaths(),name="IntensityNormalization3")
    RmICirn.inputs.op_string = "-div"
    RmICirn.inputs.out_file = "normRmIC.nii.gz"
    wf.connect(RmIC,'out_file',RmICirn,'in_file')
    wf.connect(getMaxRmIC,('out_stat',getElementFromList,1),RmICirn,"op_value")
     
    # Probability To Levelset : WM orientation
    PTLs = Node(ProbabilityToLevelset(),name='ProbaToLevelset')
    PTLs.plugin_args = {'sbatch_args':'--mem 6000'}
    PTLs.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,PTLs.name),PTLs,'output_dir')
    wf.connect(RmICirn,'out_file',PTLs,'probability_image')
     
    # Recursive Ridge Diffusion : PVS in WM only
    RRD = Node(RecursiveRidgeDiffusion(),name='RecursiveRidgeDiffusion')
    RRD.plugin_args = {'sbatch_args':'--mem 6000'}
    RRD.inputs.ridge_intensities = "bright"
    RRD.inputs.ridge_filter = "1D"
    RRD.inputs.orientation = "orthogonal"
    RRD.inputs.ang_factor = 1.0
    RRD.inputs.nb_scales = 2
    RRD.inputs.propagation_model = "diffusion"
    RRD.inputs.diffusion_factor = 1.0
    RRD.inputs.similarity_scale = 1.0
    RRD.inputs.neighborhood_size = 2
    RRD.inputs.max_iter = 100
    RRD.inputs.max_diff = 0.001
    RRD.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,RRD.name),RRD,'output_dir')
    wf.connect(ERC,'background_proba',RRD,'input_image')
    wf.connect(PTLs,'levelset',RRD,'surface_levelset')
    wf.connect(RmICirn,'out_file',RRD,'loc_prior')
    
    # Extract Lesions : extract WM PVS
    WMPVS = Node(LesionExtraction(),name='ExtractPVSfromWM')
    WMPVS.plugin_args = {'sbatch_args':'--mem 6000'}
    WMPVS.inputs.gm_boundary_partial_vol_dist = 1.0
    WMPVS.inputs.csf_boundary_partial_vol_dist = 3.0
    WMPVS.inputs.lesion_clust_dist = 1.0
    WMPVS.inputs.prob_min_thresh = 0.1
    WMPVS.inputs.prob_max_thresh = 0.33
    WMPVS.inputs.small_lesion_size = 4.0
    WMPVS.inputs.save_data = True
    WMPVS.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,WMPVS.name),WMPVS,'output_dir')
    wf.connect(RRD,'propagation',WMPVS,'probability_image')
    wf.connect(MGDM,'segmentation',WMPVS,'segmentation_image')
    wf.connect(MGDM,'distance',WMPVS,'levelset_boundary_image')
    wf.connect(RmICirn,'out_file',WMPVS,'location_prior_image')
    
    
    # Image calculator : internal capsule witout ventricules
    ICwoVent = Node(ImageMaths(), name="ICWithoutVentricules")
    ICwoVent.inputs.op_string = "-sub"
    ICwoVent.inputs.out_file = "icwovent.nii.gz"
    wf.connect(DMRP,"internal_capsule_pv",ICwoVent,"in_file")
    wf.connect(DMRP,"inter_ventricular_pv", ICwoVent, "in_file2")
    
    # Probability To Levelset : IC orientation
    PTLIC = Node(ProbabilityToLevelset(),name='ICOrientation')
    PTLIC.plugin_args = {'sbatch_args':'--mem 6000'}
    PTLIC.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,PTLIC.name),PTLIC,'output_dir')
    wf.connect(ICwoVent,'out_file',PTLIC,'probability_image')
    
    # Recursive Ridge Diffusion : PVS in IC only
    RRD2 = Node(RecursiveRidgeDiffusion(),name='RecursiveRidgeDiffusion2')
    RRD2.plugin_args = {'sbatch_args':'--mem 6000'}
    RRD2.inputs.ridge_intensities = "bright"
    RRD2.inputs.ridge_filter = "1D"
    RRD2.inputs.orientation = "undefined"
    RRD2.inputs.ang_factor = 1.0
    RRD2.inputs.nb_scales = 3
    RRD2.inputs.propagation_model = "diffusion"
    RRD2.inputs.diffusion_factor = 1.0
    RRD2.inputs.similarity_scale = 1.0
    RRD2.inputs.neighborhood_size = 2
    RRD2.inputs.max_iter = 100
    RRD2.inputs.max_diff = 0.001
    RRD2.inputs.save_data = True
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,RRD2.name),RRD2,'output_dir')
    wf.connect(ERC,'background_proba',RRD2,'input_image')
    wf.connect(PTLIC,'levelset',RRD2,'surface_levelset')
    wf.connect(DMRP,'internal_capsule_pv',RRD2,'loc_prior')
    
    # Extract Lesions : extract IC PVS
    ICPVS = Node(LesionExtraction(),name='ExtractPVSfromIC')
    ICPVS.plugin_args = {'sbatch_args':'--mem 6000'}
    ICPVS.inputs.gm_boundary_partial_vol_dist = 1.0
    ICPVS.inputs.csf_boundary_partial_vol_dist = 4.0
    ICPVS.inputs.lesion_clust_dist = 1.0
    ICPVS.inputs.prob_min_thresh = 0.25
    ICPVS.inputs.prob_max_thresh = 0.5
    ICPVS.inputs.small_lesion_size = 4.0
    ICPVS.inputs.save_data = True
    ICPVS.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,ICPVS.name),ICPVS,'output_dir')
    wf.connect(RRD2,'propagation',ICPVS,'probability_image')
    wf.connect(MGDM,'segmentation',ICPVS,'segmentation_image')
    wf.connect(MGDM,'distance',ICPVS,'levelset_boundary_image')
    wf.connect(DMRP,'internal_capsule_pv',ICPVS,'location_prior_image')
    
    
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
    wf.connect(VentHorns,"out_file", AddVentHorns, "in_file2")
    
    
    # Intensity Range Normalization (4)
    getMaxAddVentHorns = Node(ImageStats(op_string= '-r'), name="getMaxAddVentHorns")
    wf.connect(AddVentHorns,'out_file',getMaxAddVentHorns,'in_file')
     
    AddVentHornsirn = Node(AbcImageMaths(),name="IntensityNormalization4")
    AddVentHornsirn.inputs.op_string = "-div"
    AddVentHornsirn.inputs.out_file = "normAddVentHorns.nii.gz"
    wf.connect(AddVentHorns,'out_file',AddVentHornsirn,'in_file')
    wf.connect(getMaxAddVentHorns,('out_stat',getElementFromList,1),AddVentHornsirn,"op_value")
    
    # Extract Lesions : extract White Matter Hyperintensities
    WMH = Node(LesionExtraction(),name='ExtractWMH')
    WMH.plugin_args = {'sbatch_args':'--mem 6000'}
    WMH.inputs.gm_boundary_partial_vol_dist = 1.0
    WMH.inputs.csf_boundary_partial_vol_dist = 2.0
    WMH.inputs.lesion_clust_dist = 1.0
    WMH.inputs.prob_min_thresh = 0.84
    WMH.inputs.prob_max_thresh = 0.84
    WMH.inputs.small_lesion_size = 4.0
    WMH.inputs.save_data = True
    WMH.inputs.atlas_file = atlas
    wf.connect(subjectList,('subject_id',createOutputDir,wf.base_dir,wf.name,WMH.name),WMH,'output_dir')
    wf.connect(ERC2,'background_proba',WMH,'probability_image')
    wf.connect(MGDM,'segmentation',WMH,'segmentation_image')
    wf.connect(MGDM,'distance',WMH,'levelset_boundary_image')
    wf.connect(AddVentHornsirn,'out_file',WMH,'location_prior_image')
    
    
    
    
    
    
    
    ## Outputs
    
    WMPVSoutput=Node(IdentityInterface(fields=['region','lesion_size','lesion_proba','boundary','label','score']), name='WMPVSoutput')
    wf.connect(WMPVS,'lesion_prior',WMPVSoutput,'region')
    wf.connect(WMPVS,'lesion_size',WMPVSoutput,'lesion_size')
    wf.connect(WMPVS,'lesion_proba',WMPVSoutput,'lesion_proba')
    wf.connect(WMPVS,'lesion_pv',WMPVSoutput,'boundary')
    wf.connect(WMPVS,'lesion_labels',WMPVSoutput,'label')
    wf.connect(WMPVS,'lesion_score',WMPVSoutput,'score')
    
    ICPVSoutput=Node(IdentityInterface(fields=['lesion_prior','lesion_size','lesion_proba','boundary','label','score']), name='ICPVSoutput')
    wf.connect(ICPVS,'lesion_prior',ICPVSoutput,'lesion_prior')
    wf.connect(ICPVS,'lesion_size',ICPVSoutput,'lesion_size')
    wf.connect(ICPVS,'lesion_proba',ICPVSoutput,'lesion_proba')
    wf.connect(ICPVS,'lesion_pv',ICPVSoutput,'boundary')
    wf.connect(ICPVS,'lesion_labels',ICPVSoutput,'label')
    wf.connect(ICPVS,'lesion_score',ICPVSoutput,'score')
    
    WMHoutput=Node(IdentityInterface(fields=['lesion_prior','lesion_size','lesion_proba','boundary','label','score']), name='WMHoutput')
    wf.connect(WMH,'lesion_prior',WMHoutput,'lesion_prior')
    wf.connect(WMH,'lesion_size',WMHoutput,'lesion_size')
    wf.connect(WMH,'lesion_proba',WMHoutput,'lesion_proba')
    wf.connect(WMH,'lesion_pv',WMHoutput,'boundary')
    wf.connect(WMH,'lesion_labels',WMHoutput,'label')
    wf.connect(WMH,'lesion_score',WMHoutput,'score')

    return wf
    
