import numpy as np

def crop_images(src, target_shape=(640, 640), step=0.5):
    '''
    src: shape [n, c, h, w]
    '''
    H, W = src.shape[-2:]
    h_, w_ = target_shape
    step_pix = [target_i * step for target_i in target_shape]
    h_slices = int(np.ceil((H - h_) / step_pix[0]) + 1)
    w_slices = int(np.ceil((W - w_) / step_pix[1]) + 1)
    img_slices = []
    offset_slices = []
    for h_i in range(h_slices):
        for w_i in range(w_slices):
            dr_h = np.clip(h_i * h_ + h_, 0, H)
            dr_w = np.clip(w_i * w_ + w_, 0, W)
            ul = (dr_h - h_, dr_w - w_)
            img_slices.append(src[..., ul[0]:dr_h, ul[1]:dr_w])
            offset_slices.append(ul)
    return offset_slices, img_slices

