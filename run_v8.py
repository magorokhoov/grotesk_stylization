# COPYRIGHT GOROKHOV MIKHAIL ANTONOVICH 2023
# КОПИРАЙТ ГОРОХОВ МИХАИЛ АНТОНОВИЧ 2023
# My Github: https://github.com/magorokhoov

import argparse
import os.path as osp
import torch

import os
import cv2
import numpy as np
import time

from architectures.Stylish_GV8_arch import Stylish_GV8
from architectures.CoolGen import CoolGen_V6, CoolGen_V7
from architectures.CoolGen_V8B import CoolGen_V8B
from architectures.u2net import U2NET

def get_img_modcrop(img, mod):
    img_res = img.copy()
    
    height, width, _ = img_res.shape

    h_remain = height%mod
    w_remain = width%mod
    if h_remain != 0 or w_remain != 0:
        h0_pad = 0 # h_remain//2 + h_remain%2
        h1_pad = h_remain # h_remain//2
        w0_pad = w_remain//2 + w_remain%2
        w1_pad = w_remain//2

        img_res = img_res[
            h0_pad:height-h1_pad,
            w0_pad:width-w1_pad,
            ]

    return img_res

def get_img_min_side(img, min_side):
    img_res = img.copy()

    height, width, _ = img_res.shape

    if height < min_side or width < min_side:
        if height < width:
            h_new = min_side
            w_new = round(h_new*width/height)
        else:
            w_new = min_side
            h_new = round(w_new*height/width)

        img_res = cv2.resize(img_res, dsize=(w_new, h_new), interpolation=cv2.INTER_LINEAR)
        if len(img_res.shape) == 2:
            img_res = np.expand_dims(img_res, axis=2)

    return img_res

def get_img_max_square(img, max_square):
    img_res = img.copy()

    height, width, _ = img_res.shape

    if height*width > max_square:
        k = (max_square/(height*width))**0.5

        h_new = round(k*height)
        w_new = round(k*width)

        img_res = cv2.resize(img_res, dsize=(w_new, h_new), interpolation=cv2.INTER_LINEAR)
        if len(img_res.shape) == 2:
            img_res = np.expand_dims(img_res, axis=2)

    return img_res

def get_img4back_removal(img):
    # norm
    tmp_img = img.copy()
    tmp_img = cv2.resize(img, dsize=(320,320), interpolation=cv2.INTER_LINEAR)
    tmp_img = tmp_img / np.max(tmp_img)
    
    # BGR to RGB and norm
    res_img = np.zeros_like(tmp_img)
    
    res_img[:,:,0] = (tmp_img[:,:,2] - 0.485) / 0.229
    res_img[:,:,1] = (tmp_img[:,:,1] - 0.456) / 0.224
    res_img[:,:,2] = (tmp_img[:,:,0] - 0.406) / 0.225

    return res_img

def get_normed_mask(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def np2torch(x0):
    x = np.transpose(x0, (2,0,1)) # from hwc to chw
    x = np.expand_dims(x, axis=0) # add batch
    x = torch.from_numpy(x) # from np to torch
    
    return x

def torch2np(x0):
    x = x0.numpy()
    x = x[0] # remove batch
    x = np.transpose(x, (1,2,0))

    return x
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-models', '-m', type=str, required=True, help='Path to models.')
    parser.add_argument('-autocrop', type=str, required=True, default='yes', help='autocrop after u2net background remove (yes/no)')
    parser.add_argument('-input', '-i', type=str, required=False, default='./input', help='Path to read input images.')
    parser.add_argument('-output', '-o', type=str, required=False, default='./output', help='Path to save output images.')
    parser.add_argument('-no_gpu', '-cpu', required=False, action='store_false', help='Run in CPU if enabled.')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    gpu = args.no_gpu  # TODO: fp16 error with cpu: RuntimeError: "unfolded2d_copy" not implemented for 'Half'

    device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu') 

    print(device)

    input_dir = args.input
    output_dir = args.output

    model_back_removal = U2NET(in_ch=3, out_ch=1)


    params = {
	    'in_nc': 2,
		'base_nc': 64,
		'out_nc': 1,
        'num_truck2_blocks': 8,
        'num_truck3_blocks': 12,
        'num_truck4_blocks': 16,
        'num_truck5_blocks': 16,
        'attention_type': 'srm',
    }

    model_stylish = CoolGen_V8B(option_arch=params)

    model_back_removal.to(device)
    model_back_removal.eval()

    model_stylish.to(device)
    model_stylish.eval()

    path_stylish = args.models
    model_stylish.load_state_dict(torch.load(path_stylish))
    model_back_removal.load_state_dict(torch.load('./models/u2net_human_seg.pth', map_location=torch.device('cpu')))

    images_names = sorted(os.listdir(input_dir))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for image_name in images_names:


# COPYRIGHT GOROKHOV MIKHAIL ANTONOVICH 2022
# КОПИРАЙТ ГОРОХОВ МИХАИЛ АНТОНОВИЧ 2022
# My Github: https://github.com/magorokhoov

        # step 0: read img, norm # 
        image_path = os.path.join(input_dir, image_name)

        image_name = osp.splitext(osp.basename(image_path))[0]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        print(f'Inferencing: {image_name}')
        if img is None:
            print(f'Error reading image {image_path}, skipping.')
            continue

        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 0.0001) # [0,1] # *(255.0)

        # step 1: background removal
        img_backremoval = get_img4back_removal(img)
        img_backremoval = np2torch(img_backremoval).to(device)


        t1 = time.time()
        with torch.no_grad():
            mask = model_back_removal(img_backremoval).detach()

        mask = get_normed_mask(mask)
        mask = torch2np(mask)

        mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)


        # step 2: RGB to Gray and apply mask to img
        img = (img[:,:,0:1] + img[:,:,1:2] + img[:,:,2:3]) / 3.0
        img_wo_back = img_wo_back = np.concatenate((img, mask), axis=2)

        # step 3: autocrop
        y0 = 0
        y1 = img.shape[0]
        x0 = 0
        x1 = img.shape[1]
        if args.autocrop == 'yes':
            while y0 < img.shape[0]//2 and mask[0:y0+1].sum() < 6: # 6 pixels with 255 or sum light is 6*255 in one row
                y0 += 1
            while mask[:,0:x0+1].sum() < 6:
                x0 += 1
            while mask[:,x1-1:].sum() < 6:
                x1 -= 1
        
            # add small padding
            pad = 18
            x0 = max(0, x0-pad)
            x1 = min(img.shape[1], x1+pad)
            y0 = max(0, y0-pad)
            y1 = min(img.shape[0], y1+pad)

            
            img_wo_back = img_wo_back[y0:y1, x0:x1]
        ## end autocrop
        

        # step 4: preporation for stylization
        img_wo_back = get_img_max_square(img_wo_back, 1600*2400)
        img_wo_back = get_img_min_side(img_wo_back, 800)

        #img_wo_back = get_img_modcrop(img_wo_back, 8)

        t_out = np2torch(img_wo_back).clone()

        # step 5: it's about time stylization
        t2 = time.time()
        with torch.no_grad():
            t_out = model_stylish(t_out)
        t3 = time.time()

        # step 6: post proccesing
        img_result = torch2np(t_out.detach())

        img_result = (254*img_result).clip(0,255)
        img_result = img_result.astype(np.uint8)

        # step 7 final: save results and clear memory
        save_img_style_path = osp.join(output_dir, f'{image_name:s}.png')
        cv2.imwrite(save_img_style_path, img_result, [cv2.IMWRITE_PNG_COMPRESSION, 6])

        print(f'Mask: {round(t2-t1, 3)}s', end='\t')
        print(f'Stylization: {round(t3-t2, 3)}s', end='\t')
        print(f'Total: {round(t3-t1, 3)}s')

        del img, img_backremoval, img_wo_back, img_result, mask, t_out
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


# COPYRIGHT GOROKHOV MIKHAIL ANTONOVICH 2022
# КОПИРАЙТ ГОРОХОВ МИХАИЛ АНТОНОВИЧ 2022
# My Github: https://github.com/magorokhoov
