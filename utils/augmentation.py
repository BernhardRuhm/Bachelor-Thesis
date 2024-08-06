import numpy as np
import matplotlib.pyplot as plt

# Implementation of differnt augmentation methods are from https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

 #TODO: get number of needed samples in util function
 # also add corresponding random y
def augment_data(X, y, sample_size, augmentation_type):

    X_aug = []
    y_aug = []

    for _ in range(sample_size):
        rs = np.random.randint(len(X))
        X_aug.append(X[rs])
        y_aug.append(y[rs])

    X_aug = np.array(X_aug)

    if augmentation_type == 'jitter':
        X_aug = jitter(X_aug)
    elif augmentation_type == 'scaling':
        X_aug = scaling(X_aug)
    elif augmentation_type == 'window_warp':
        X_aug = window_warp(X_aug)
    elif augmentation_type == 'magnitude_warp':
        X_aug = magnitude_warp(X_aug)
    elif augmentation_type == 'rotation':
        X_aug = rotation(X_aug)

    
    X_aug = np.concatenate((X, X_aug))
    y_aug = np.concatenate((y, y_aug))


    # Shuffle original and augmented data
    indices = np.arange(X_aug.shape[0])
    np.random.shuffle(indices)
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]

    return X_aug, y_aug


def get_augmentation_type(data_type):
    if data_type in ["IMAGE", "SPECTRO"]: 
        return "magnitude_warp"
    elif data_type == "ECG":
        return "scaling"
    elif data_type in ["MOTION", "SIMULATED", "HAR"]:
        return "window_warp"
    elif data_type == "POWER":
        return "rotation"
    elif data_type == ["ECG", "SENSOR", "AUDIO", "DEVICE"]:
        return "jitter"

    return None
