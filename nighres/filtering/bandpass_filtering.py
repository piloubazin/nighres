import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def bandpass_filtering(time_series, repetition_time,
                       low_frequency=0.01, high_frequency=0.1,
                       save_data=False, output_dir=None,
                       file_name=None):
    """ Basic bandpass filtering

    Filters out high and low frequencies from fMRI time series 
    (the 4th dimension of a 4D image) by Fourier transform, masking,
    and Fourier inverse transform.

    Parameters
    ----------
    time_series: niimg
        Time series image (4D data)
    low_frequency: float
        Low frequency cutoff (default is 0.01 Hz)
    high_frequency: float
        High frequency cutoff (default is 0.1 Hz)
    repetition_time: float
        Time interval between samples, aka repetition time or TR
    save_data: bool
        Save output data to file (default is False)
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

        * filtered (niimg): Binary brain mask (_bpf)

    Notes
    ----------

    References
    ----------
    """

    print('\nBandpass filtering')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, second_inversion)
    
        filtered_file = _fname_4saving(file_name=file_name,
                                   rootfile=time_series,
                                   suffix='bpf')
    
     # get dimensions and resolution from second inversion image
    img = load_volume(time_series)
    data = img.get_data()
    affine = img.affine #or img.get_affine(), which I think is being deprecated?
    header = img.header
    
    length = np.shape(data)[3]
    
    print("defining the frequency window")
    nextpowerof2 = np.ceil(np.log2(length))
    padded = int(np.power(2, nextpowerof2))
    
    #print("freq:",freq,"lf:",low,"hf:",high,"length:",padded)
    if (low >= freq/2):
    	lowid = int(padded/2)
    else:
    	lowid = int(np.ceil(low*padded*tr))
    
    if (high >= freq/2):
    	highid = int(padded/2)
    else:
    	highid = int(np.floor(high*padded*tr))
    
    frequencymask = np.zeros(padded)
    frequencymask[lowid+1:highid+1] = 1
    frequencymask[padded-highid:padded-lowid] = 1
    #print(lowid,highid)
    #print(frequencymask)
    
    print("removing the mean")
    datamean = data.mean(3,keepdims=True)
    datamean = np.repeat(datamean,length,axis=3)
    
    data = np.pad(data-datamean,((0,0),(0,0),(0,0),(0,padded-length)),'constant') 
    #print("shape: ",np.shape(data))
    
    print("filtering")
    data = np.fft.fft(data,axis=3)
    #print("shape: ",np.shape(data))
    data[:,:,:,np.nonzero(frequencymask==0)] = 0
    data = np.real(np.fft.ifft(data,axis=3))
    
    data = data[:,:,:,0:length]+datamean
    
    # collect outputs and potentially save
    filtered_img = nb.Nifti1Image(data, affine, header)
    filtered_img.header['cal_min'] = np.min(data)
    filtered_img.header['cal_max'] = np.max(data)
    
    outputs = {'filtered': filtered_img}
    
    if save_data:
        save_volume(os.path.join(output_dir, filtered_file), filtered)
    
    return outputs
