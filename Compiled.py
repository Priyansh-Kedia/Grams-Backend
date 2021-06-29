#################################################################################################################################
##################################################### core.py ###################################################################
#################################################################################################################################

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2

# from . import transforms, utils, metrics

### removing because mxnet will not be used
# try:
#     from mxnet import gluon, nd
#     import mxnet as mx
#     # from . import resnet_style
#     MXNET_ENABLED = True 
#     mx_GPU = mx.gpu()
#     mx_CPU = mx.cpu()
# except:
MXNET_ENABLED = False

try:
    import torch
    from torch import optim, nn
    from torch.utils import mkldnn as mkldnn_utils
    # from . import resnet_torch
    TORCH_ENABLED = True 
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False

def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        print('parsing model string to get unet options')
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        print('parsing model string to get cellpose options')
        nclasses = 3
    else:
        return None
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0]=='on'
    style_on = ostrs[1]=='on'
    concatenation = ostrs[2]=='on'
    return nclasses, residual_on, style_on, concatenation

def use_gpu(gpu_number=0, istorch=True):
    """ check if gpu works """
    if istorch:
        return _use_gpu_torch(gpu_number)
    

# def _use_gpu_mxnet(gpu_number=0):
#     try:
#         _ = mx.ndarray.array([1, 2, 3], ctx=mx.gpu(gpu_number))
#         print('** MXNET CUDA version installed and working. **')
#         return True
#     except mx.MXNetError:
#         print('MXNET CUDA version not installed/working.')
#         return False

def _use_gpu_torch(gpu_number=0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        print('** TORCH CUDA version installed and working. **')
        return True
    except:
        print('TORCH CUDA version not installed/working.')
        return False

def assign_device(istorch, gpu):
    if gpu and use_gpu(istorch=istorch):
        device = torch_GPU if istorch else True #mx_GPU
        gpu=True
        print('>>>> using GPU')
    else:
        device = torch_CPU if istorch else True #mx_CPU
        print('>>>> using CPU')
        gpu=False
    return device, gpu

def check_mkl(istorch=True):
    print('Running test snippet to check if MKL-DNN working')
    if istorch:
        print('see https://pytorch.org/docs/stable/backends.html?highlight=mkl')
    else:
        print('see https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_readme.html#4)')
    if istorch:
        mkl_enabled = torch.backends.mkldnn.is_available()
    else:
        process = subprocess.Popen(['python', 'test_mkl.py'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    cwd=os.path.dirname(os.path.abspath(__file__)))
        stdout, stderr = process.communicate()
        if len(stdout)>0:
            mkl_enabled = True
        else:
            mkl_enabled = False
    if mkl_enabled:
        print('** MKL version working - CPU version is sped up. **')
    elif not istorch:
        print('WARNING: MKL version on mxnet not working/installed - CPU version will be SLOW.')
    else:
        print('WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
    return mkl_enabled


def convert_images(x, channels, do_3D, normalize, invert):
    """ return list of images with channels last and normalized intensities """
    if not isinstance(x,list) and not (x.ndim>3 and not do_3D):
        nolist = True
        x = [x]
    else:
        nolist = False
    
    nimg = len(x)
    if do_3D:
        for i in range(len(x)):
            if x[i].ndim<3:
                raise ValueError('ERROR: cannot process 2D images in 3D mode') 
            elif x[i].ndim<4:
                x[i] = x[i][...,np.newaxis]
            if x[i].shape[1]<4:
                x[i] = x[i].transpose((0,2,3,1))
            elif x[i].shape[0]<4:
                x[i] = x[i].transpose((1,2,3,0))
            print('multi-stack tiff read in as having %d planes %d channels'%
                    (x[i].shape[0], x[i].shape[-1]))

    if channels is not None:
        if len(channels)==2:
            if not isinstance(channels[0], list):
                channels = [channels for i in range(nimg)]
        for i in range(len(x)):
            if x[i].shape[0]<4:
                x[i] = x[i].transpose(1,2,0)
        # x = [transforms.reshape(x[i], channels=channels[i]) for i in range(nimg)]
        x = [reshape(x[i], channels=channels[i]) for i in range(nimg)]
    elif do_3D:
        for i in range(len(x)):
            # code above put channels last
            if x[i].shape[-1]>2:
                print('WARNING: more than 2 channels given, use "channels" input for specifying channels - just using first two channels to run processing')
                x[i] = x[i][...,:2]
    else:
        for i in range(len(x)):
            if x[i].ndim>3:
                raise ValueError('ERROR: cannot process 4D images in 2D mode')
            elif x[i].ndim==2:
                x[i] = np.stack((x[i], np.zeros_like(x[i])), axis=2)
            elif x[i].shape[0]<8:
                x[i] = x[i].transpose((1,2,0))
            if x[i].shape[-1]>2:
                print('WARNING: more than 2 channels given, use "channels" input for specifying channels - just using first two channels to run processing')
                x[i] = x[i][:,:,:2]

    if normalize or invert:
        # x = [transforms.normalize_img(x[i], invert=invert) for i in range(nimg)]
        x = [normalize_img(x[i], invert=invert) for i in range(nimg)]
    return x, nolist


class UnetModel():
    def __init__(self, gpu=False, pretrained_model=False,
                    diam_mean=30., net_avg=True, device=None,
                    residual_on=False, style_on=False, concatenation=True,
                    nclasses = 3, torch=True):
        self.unet = True
        if torch:
            if not TORCH_ENABLED:
                print('torch not installed')
                torch = False
        self.torch = torch
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        if torch and not self.gpu:
            self.mkldnn = check_mkl(self.torch)
        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        
        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                                ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])                                             
        if pretrained_model:
            print(self.net_type)
        # create network
        self.nclasses = nclasses
        nbase = [32,64,128,256]
        if self.torch:
            nchan = 2
            nbase = [nchan, 32, 64, 128, 256]
            # self.net = resnet_torch.CPnet(nbase, self.nclasses, 3, residual_on=residual_on, style_on=style_on, concatenation=concatenation, mkldnn=self.mkldnn).to(self.device)
            self.net = rtCPnet(nbase, self.nclasses, 3, residual_on=residual_on, style_on=style_on, concatenation=concatenation, mkldnn=self.mkldnn).to(self.device)
        # else:
        #     # self.net = resnet_style.CPnet(nbase, nout=self.nclasses, residual_on=residual_on, style_on=style_on, concatenation=concatenation)
        #     self.net = rsCPnet(nbase, nout=self.nclasses, residual_on=residual_on, style_on=style_on, concatenation=concatenation)
        #     self.net.hybridize(static_alloc=True, static_shape=True)
        #     self.net.initialize(ctx = self.device)

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_model(pretrained_model, cpu=(not self.gpu))

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True,
             rescale=None, do_3D=False, anisotropy=None, net_avg=True, augment=False,
             tile=True, cell_threshold=None, boundary_threshold=None, min_size=15):
        """ segment list of images x

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            cell_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            boundary_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        x, nolist = convert_images(x, channels, do_3D, normalize, invert)
        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        if nimg > 1:
            iterator = trange(nimg)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list):
            model_path = self.pretrained_model[0]
            if not net_avg:
                self.net.load_model(self.pretrained_model[0])
                if not self.torch:
                    self.net.collect_params().grad_req = 'null'
        else:
            model_path = self.pretrained_model

        if cell_threshold is None or boundary_threshold is None:
            try:
                thresholds = np.load(model_path+'_cell_boundary_threshold.npy')
                cell_threshold, boundary_threshold = thresholds
                print('>>>> found saved thresholds from validation set')
            except:
                print('WARNING: no thresholds found, using default / user input')

        cell_threshold = 2.0 if cell_threshold is None else cell_threshold
        boundary_threshold = 0.5 if boundary_threshold is None else boundary_threshold

        if not do_3D:
            for i in iterator:
                img = x[i].copy()
                shape = img.shape
                # rescale image for flow computation
                # imgs = transforms.resize_image(img, rsz=rescale[i])
                imgs = resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg, augment=augment, 
                                          tile=tile)
                
                # maski = utils.get_masks_unet(y, cell_threshold, boundary_threshold)
                maski = get_masks_unet(y, cell_threshold, boundary_threshold)
                # maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                maski = fill_holes_and_remove_small_masks(maski, min_size=min_size)
                # maski = transforms.resize_image(maski, shape[-3], shape[-2], interpolation=cv2.INTER_NEAREST)
                maski = resize_image(maski, shape[-3], shape[-2], interpolation=cv2.INTER_NEAREST)
                masks.append(maski)
                styles.append(style)
        else:
            for i in iterator:
                tic=time.time()
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile)
                yf = yf.mean(axis=0)
                print('probabilities computed %2.2fs'%(time.time()-tic))
                # maski = utils.get_masks_unet(yf.transpose((1,2,3,0)), cell_threshold, boundary_threshold)
                maski = get_masks_unet(yf.transpose((1,2,3,0)), cell_threshold, boundary_threshold)
                # maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                maski = fill_holes_and_remove_small_masks(maski, min_size=min_size)
                masks.append(maski)
                styles.append(style)
                print('masks computed %2.2fs'%(time.time()-tic))
                flows.append(yf)

        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]

        return masks, flows, styles

    def _to_device(self, x):
        if self.torch:
            X = torch.from_numpy(x).float().to(self.device)
        # else:
        #     #if x.dtype != 'bool':
        #     X = nd.array(x.astype(np.float32), ctx=self.device)
        return X

    def _from_device(self, X):
        if self.torch:
            x = X.detach().cpu().numpy()
        else:
            x = X.asnumpy()
        return x

    def network(self, x, return_conv=False):
        """ convert imgs to torch/mxnet and run network model and return numpy """
        X = self._to_device(x)
        if self.torch:
            self.net.eval()
            if self.mkldnn:
                self.net = mkldnn_utils.to_mkldnn(self.net)
        y, style, conv = self.net(X)
        if self.mkldnn:
            self.net.to(torch_CPU)
        y = self._from_device(y)
        style = self._from_device(style)
        if return_conv:
            conv = self._from_device(conv)
            y = np.concatenate((y, conv), axis=1)
        
        return y, style
                
    def _run_nets(self, img, net_avg=True, augment=False, tile=True, tile_overlap=0.1, bsize=224, 
                  return_conv=False, progress=None):
        """ run network (if more than one, loop over networks and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

        Returns
        ------------------

        y: array [3 x Ly x Lx] or [3 x Lz x Ly x Lx]
            y is output (averaged over networks);
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles,
            but not averaged over networks.

        """
        if isinstance(self.pretrained_model, str) or not net_avg:  
            y, style = self._run_net(img, augment=augment, tile=tile, 
                                     bsize=bsize, return_conv=return_conv)
        else:  
            for j in range(len(self.pretrained_model)):
                self.net.load_model(self.pretrained_model[j], cpu=(not self.gpu))
                if not self.torch:
                    self.net.collect_params().grad_req = 'null'
                y0, style = self._run_net(img, augment=augment, tile=tile, 
                                          tile_overlap=tile_overlap, bsize=bsize,
                                          return_conv=return_conv)

                if j==0:
                    y = y0
                else:
                    y += y0
                if progress is not None:
                    progress.setValue(10 + 10*j)
            y = y / len(self.pretrained_model)
        return y, style

    def _run_net(self, imgs, augment=False, tile=True, tile_overlap=0.1, bsize=224,
                 return_conv=False):
        """ run network on image or stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """   
        if imgs.ndim==4:  
            # make image Lz x nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (0,3,1,2)) 
            detranspose = (0,2,3,1)
            return_conv = False
        else:
            # make image nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (2,0,1))
            detranspose = (1,2,0)

        # pad image for net so Ly and Lx are divisible by 4
        # imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        imgs, ysub, xsub = pad_image_ND(imgs)
        # slices from padding
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-3] = slice(0, self.nclasses + 32*return_conv + 1)
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # run network
        if tile or augment or imgs.ndim==4:
            y, style = self._run_tiled(imgs, augment=augment, bsize=bsize, 
                                      tile_overlap=tile_overlap, 
                                      return_conv=return_conv)
        else:
            imgs = np.expand_dims(imgs, axis=0)
            y, style = self.network(imgs, return_conv=return_conv)
            y, style = y[0], style[0]
        style /= (style**2).sum()**0.5

        # slice out padding
        y = y[slc]

        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)
         
        return y, style
    
    def _run_tiled(self, imgi, augment=False, bsize=224, tile_overlap=0.1, return_conv=False):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
         
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        Returns
        ------------------

        yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles

        """

        if imgi.ndim==4:
            batch_size = self.batch_size 
            Lz, nchan = imgi.shape[:2]
            # IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize, augment=augment, tile_overlap=tile_overlap)
            IMG, ysub, xsub, Ly, Lx = make_tiles(imgi[0], bsize=bsize, augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            batch_size *= max(4, (bsize**2 // (ly*lx))**0.5)
            yf = np.zeros((Lz, self.nclasses, imgi.shape[-2], imgi.shape[-1]), np.float32)
            styles = []
            if ny*nx > batch_size:
                ziterator = trange(Lz)
                for i in ziterator:
                    yfi, stylei = self._run_tiled(imgi[i], augment=augment, 
                                                  bsize=bsize, tile_overlap=tile_overlap)
                    yf[i] = yfi
                    styles.append(stylei)
            else:
                # run multiple slices at the same time
                ntiles = ny*nx
                nimgs = max(2, int(np.round(batch_size / ntiles)))
                niter = int(np.ceil(Lz/nimgs))
                ziterator = trange(niter)
                for k in ziterator:
                    IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        # IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize, augment=augment, tile_overlap=tile_overlap)
                        IMG, ysub, xsub, Ly, Lx = make_tiles(imgi[k*nimgs+i], bsize=bsize, augment=augment, tile_overlap=tile_overlap)
                        IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                    ya, stylea = self.network(IMGa)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        y = ya[i*ntiles:(i+1)*ntiles]
                        if augment:
                            y = np.reshape(y, (ny, nx, 3, ly, lx))
                            # y = transforms.unaugment_tiles(y, self.unet)
                            y = unaugment_tiles(y, self.unet)
                            y = np.reshape(y, (-1, 3, ly, lx))
                        # yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                        yfi = average_tiles(y, ysub, xsub, Ly, Lx)
                        yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                        yf[k*nimgs+i] = yfi
                        stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                        stylei /= (stylei**2).sum()**0.5
                        styles.append(stylei)
            return yf, np.array(styles)
        else:
            # IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, augment=augment, tile_overlap=tile_overlap)
            IMG, ysub, xsub, Ly, Lx = make_tiles(imgi, bsize=bsize, augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            batch_size = self.batch_size
            niter = int(np.ceil(IMG.shape[0] / batch_size))
            nout = self.nclasses + 32*return_conv
            y = np.zeros((IMG.shape[0], nout, ly, lx))
            for k in range(niter):
                irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                y0, style = self.network(IMG[irange], return_conv=return_conv)
                y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
                if k==0:
                    styles = style[0]
                styles += style.sum(axis=0)
            styles /= IMG.shape[0]
            if augment:
                y = np.reshape(y, (ny, nx, nout, bsize, bsize))
                # y = transforms.unaugment_tiles(y, self.unet)
                y = unaugment_tiles(y, self.unet)
                y = np.reshape(y, (-1, nout, bsize, bsize))
            
            # yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
            yf = average_tiles(y, ysub, xsub, Ly, Lx)
            yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
            styles /= (styles**2).sum()**0.5
            return yf, styles

    def _run_3D(self, imgs, rsz=1.0, anisotropy=None, net_avg=True, 
                augment=False, tile=True, tile_overlap=0.1, 
                bsize=224, progress=None):
        """ run network on stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI


        Returns
        ------------------

        yf: array [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """ 
        sstr = ['YX', 'ZY', 'ZX']
        if anisotropy is not None:
            rescaling = [[rsz, rsz],
                         [rsz*anisotropy, rsz],
                         [rsz*anisotropy, rsz]]
        else:
            rescaling = [rsz] * 3
        pm = [(0,1,2,3), (1,0,2,3), (2,0,1,3)]
        ipm = [(3,0,1,2), (3,1,0,2), (3,1,2,0)]
        yf = np.zeros((3, self.nclasses, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
        for p in range(3 - 2*self.unet):
            xsl = imgs.copy().transpose(pm[p])
            # rescale image for flow computation
            shape = xsl.shape
            # xsl = transforms.resize_image(xsl, rsz=rescaling[p])    
            xsl = resize_image(xsl, rsz=rescaling[p])    
            # per image
            print('\n running %s: %d planes of size (%d, %d) \n\n'%(sstr[p], shape[0], shape[1], shape[2]))
            y, style = self._run_nets(xsl, net_avg=net_avg, augment=augment, tile=tile, 
                                      bsize=bsize, tile_overlap=tile_overlap)
            # y = transforms.resize_image(y, shape[1], shape[2])    
            y = resize_image(y, shape[1], shape[2])    
            yf[p] = y.transpose(ipm[p])
            if progress is not None:
                progress.setValue(25+15*p)
        return yf, style

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        # if available set boundary pixels to 2
        if lbl.shape[1]>1 and self.nclasses>2:
            boundary = lbl[:,1]<=4
            lbl = lbl[:,0]
            lbl[boundary] *= 2
        else:
            lbl = lbl[:,0]
        lbl = self._to_device(lbl)
        loss = 8 * 1./self.nclasses * self.criterion(y, lbl)
        return loss

    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None, save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, rescale=False):
        """ train function uses 0-1 mask label and boundary pixels for training """

        nimg = len(train_data)

        # train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize)
        train_data, train_labels, test_data, test_labels, run_test = reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize)

        # add dist_to_bound to labels
        if self.nclasses==3:
            print('computing boundary pixels')
            # train_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
            train_classes = [np.stack((label, label>0, distance_to_boundary(label)), axis=0).astype(np.float32)
                                for label in tqdm(train_labels)]
        else:
            train_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                for label in tqdm(train_labels)]
        if run_test:
            if self.nclasses==3:
                # test_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                test_classes = [np.stack((label, label>0, distance_to_boundary(label)), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels)]
            else:
                test_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels)]
        
        # split train data into train and val
        val_data = train_data[::8]
        val_classes = train_classes[::8]
        val_labels = train_labels[::8]
        del train_data[::8], train_classes[::8], train_labels[::8]

        model_path = self._train_net(train_data, train_classes, 
                                     test_data, test_classes,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, momentum, weight_decay, 
                                     batch_size, rescale)


        # find threshold using validation set
        print('>>>> finding best thresholds using validation set')
        cell_threshold, boundary_threshold = self.threshold_validation(val_data, val_labels)
        np.save(model_path+'_cell_boundary_threshold.npy', np.array([cell_threshold, boundary_threshold]))

    def threshold_validation(self, val_data, val_labels):
        cell_thresholds = np.arange(-4.0, 4.25, 0.5)
        if self.nclasses==3:
            boundary_thresholds = np.arange(-2, 2.25, 1.0)
        else:
            boundary_thresholds = np.zeros(1)
        aps = np.zeros((cell_thresholds.size, boundary_thresholds.size, 3))
        for j,cell_threshold in enumerate(cell_thresholds):
            for k,boundary_threshold in enumerate(boundary_thresholds):
                masks = []
                for i in range(len(val_data)):
                    output,style = self._run_net(val_data[i].transpose(1,2,0), augment=False)
                    # masks.append(utils.get_masks_unet(output, cell_threshold, boundary_threshold))
                    masks.append(get_masks_unet(output, cell_threshold, boundary_threshold))
                # ap = metrics.average_precision(val_labels, masks)[0]
                ap = average_precision(val_labels, masks)[0]
                ap0 = ap.mean(axis=0)
                aps[j,k] = ap0
            if self.nclasses==3:
                kbest = aps[j].mean(axis=-1).argmax()
            else:
                kbest = 0
            if j%4==0:
                print('best threshold at cell_threshold = {} => boundary_threshold = {}, ap @ 0.5 = {}'.format(cell_threshold, boundary_thresholds[kbest], 
                                                                        aps[j,kbest,0]))   
        if self.nclasses==3: 
            jbest, kbest = np.unravel_index(aps.mean(axis=-1).argmax(), aps.shape[:2])
        else:
            jbest = aps.squeeze().mean(axis=-1).argmax()
            kbest = 0
        cell_threshold, boundary_threshold = cell_thresholds[jbest], boundary_thresholds[kbest]
        print('>>>> best overall thresholds: (cell_threshold = {}, boundary_threshold = {}); ap @ 0.5 = {}'.format(cell_threshold, boundary_threshold, 
                                                          aps[jbest,kbest,0]))
        return cell_threshold, boundary_threshold

    def _train_step(self, x, lbl):
        X = self._to_device(x)
        if self.torch:
            self.optimizer.zero_grad()
            if self.gpu:
                self.net.train().cuda()
            else:
                self.net.train()
            y = self.net(X)[0]
            loss = self.loss_fn(lbl,y)
            loss.backward()
            train_loss = loss.item()
            self.optimizer.step()
            train_loss *= len(x)
        # else:
        #     with mx.autograd.record():
        #         y = self.net(X)[0]
        #         loss = self.loss_fn(lbl, y)
        #     loss.backward()
        #     train_loss = nd.sum(loss).asscalar()
        #     self.optimizer.step(x.shape[0])
        return train_loss

    def _test_eval(self, x, lbl):
        X = self._to_device(x)
        if self.torch:
            self.net.eval()
            y, style = self.net(X)
            loss = self.loss_fn(lbl,y)
            test_loss = loss.item()
            test_loss *= len(x)
        # else:
        #     y, style = self.net(X)
        #     loss = self.loss_fn(lbl, y)
        #     test_loss = nd.sum(loss).asnumpy()
        return test_loss

    def _set_optimizer(self, learning_rate, momentum, weight_decay):
        if self.torch:
            self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate,
                            momentum=momentum, weight_decay=weight_decay)
        # else:
        #     self.optimizer = gluon.Trainer(self.net.collect_params(), 'sgd',{'learning_rate': learning_rate,
        #                         'momentum': momentum, 'wd': weight_decay})

    def _set_learning_rate(self, lr):
        if self.torch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.optimizer.set_learning_rate(lr)

    def _set_criterion(self):
        if self.unet:
            if self.torch:
                criterion = nn.SoftmaxCrossEntropyLoss(axis=1)
            # else:
            #     criterion = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
        else:
            if self.torch:
                self.criterion  = nn.MSELoss(reduction='mean')
                self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
            # else:
            #     self.criterion  = gluon.loss.L2Loss()
            #     self.criterion2 = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    def _train_net(self, train_data, train_labels, 
              test_data=None, test_labels=None,
              pretrained_model=None, save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, 
              batch_size=8, rescale=True, netstr='cellpose'):
        """ train function uses loss function self.loss_fn """

        d = datetime.datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self._set_optimizer(self.learning_rate, momentum, weight_decay)
        self._set_criterion()
        

        nimg = len(train_data)

        # compute average cell diameter
        if rescale:
            # diam_train = np.array([utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
            diam_train = np.array([diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
            diam_train[diam_train<5] = 5.
            if test_data is not None:
                # diam_test = np.array([utils.diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test = np.array([diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test[diam_test<5] = 5.
            scale_range = 0.5
        else:
            scale_range = 1.0

        nchan = train_data[0].shape[0]
        print('>>>> training network with %d channel input <<<<'%nchan)
        print('>>>> saving every %d epochs'%save_every)
        print('>>>> median diameter = %d'%self.diam_mean)
        print('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate, self.batch_size, weight_decay))
        print('>>>> ntrain = %d'%nimg)
        if test_data is not None:
            print('>>>> ntest = %d'%len(test_data))
        print(train_data[0].shape)

        # set learning rate schedule    
        LR = np.linspace(0, self.learning_rate, 10)
        if self.n_epochs > 250:
            LR = np.append(LR, self.learning_rate*np.ones(self.n_epochs-100))
            for i in range(10):
                LR = np.append(LR, LR[-1]/2 * np.ones(10))
        else:
            LR = np.append(LR, self.learning_rate*np.ones(max(0,self.n_epochs-10)))
        
        tic = time.time()

        lavg, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            print('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        # cannot train with mkldnn
        self.net.mkldnn = False

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)
            self._set_learning_rate(LR[iepoch])

            for ibatch in range(0,nimg,batch_size):
                inds = rperm[ibatch:ibatch+batch_size]
                rsc = diam_train[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
                # imgi, lbl, scale = transforms.random_rotate_and_resize(
                imgi, lbl, scale = random_rotate_and_resize(
                                        [train_data[i] for i in inds], Y=[train_labels[i][1:] for i in inds],
                                        rescale=rsc, scale_range=scale_range, unet=self.unet)
                # if self.unet and lbl.shape[1]>1 and rescale:
                #     lbl[:,1] /= diam_batch[:,np.newaxis,np.newaxis]**2
                train_loss = self._train_step(imgi, lbl)
                lavg += train_loss
                nsum += len(imgi) 
            
            if iepoch%10==0 or iepoch<10:
                lavg = lavg / nsum
                if test_data is not None:
                    lavgt, nsum = 0., 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    for ibatch in range(0,len(test_data),batch_size):
                        inds = rperm[ibatch:ibatch+batch_size]
                        rsc = diam_test[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
                        # imgi, lbl, scale = transforms.random_rotate_and_resize(
                        imgi, lbl, scale = random_rotate_and_resize(
                                            [test_data[i] for i in inds],
                                            Y=[test_labels[i][1:] for i in inds],
                                            scale_range=0., rescale=rsc, unet=self.unet)
                        if self.unet and lbl.shape[1]>1 and rescale:
                            lbl[:,1] *= scale[0]**2

                        test_loss = self._test_eval(imgi, lbl)
                        lavgt += test_loss
                        nsum += len(imgi)

                    print('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, lavgt/nsum, LR[iepoch]))
                else:
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, LR[iepoch]))
                lavg, nsum = 0, 0

            if save_path is not None:
                if iepoch==self.n_epochs-1 or iepoch%save_every==1:
                    # save model at the end
                    file = '{}_{}_{}'.format(self.net_type, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                    ksave += 1
                    print('saving network parameters')
                    self.net.save_model(os.path.join(file_path, file))

        # reset to mkldnn if available
        self.net.mkldnn = self.mkldnn

        return os.path.join(file_path, file)


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### dynamics.py ###############################################################
#################################################################################################################################

import time, os
from scipy.ndimage.filters import maximum_filter1d
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
from numba import njit, float32, int32, vectorize
# from . import utils, metrics

from njitFunc import steps2D, steps3D, map_coordinates, _extend_centers

try:
    import torch
    from torch import optim, nn
    # from . import resnet_torch
    TORCH_ENABLED = True 
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False

# #@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
# def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
#     """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)

#     Parameters
#     --------------

#     T: float64, array
#         _ x Lx array that diffusion is run in

#     y: int32, array
#         pixels in y inside mask

#     x: int32, array
#         pixels in x inside mask

#     ymed: int32
#         center of mask in y

#     xmed: int32
#         center of mask in x

#     Lx: int32
#         size of x-dimension of masks

#     niter: int32
#         number of iterations to run diffusion

#     Returns
#     ---------------

#     T: float64, array
#         amount of diffused particles at each pixel

#     """

#     for t in range(niter):
#         T[ymed*Lx + xmed] += 1
#         T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
#                                             T[y*Lx + x-1]     + T[y*Lx + x+1] +
#                                             T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
#                                             T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
#     return T

tic=time.time()

def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    neighbors is 9 x pixels in masks, 
    centers are mask centers, 
    isneighbor is valid neighbor boolean 9 x pixels
    
    """
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device)
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:,0], meds[:,1]] +=1
        Tneigh = T[:, pt[:,:,0], pt[:,:,1]]
        Tneigh *= isneigh
        T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1)
  
    T = torch.log(1.+ T)
    # gradient positions
    grads = T[:, pt[[2,1,4,3],:,0], pt[[2,1,4,3],:,1]]
    dy = grads[:,0] - grads[:,1]
    dx = grads[:,2] - grads[:,3]

    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch


def masks_to_flows_gpu(masks, device=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined using COM

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if device is None:
        device = torch.device('cuda')

    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2, Lx0+2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    centers = np.array(scipy.ndimage.center_of_mass(masks_padded, labels=masks_padded, 
                                                    index=np.arange(1, masks_padded.max()+1))).astype(int)
    # (check mask center inside mask)
    valid = masks_padded[centers[:,0], centers[:,1]] == np.arange(1, masks_padded.max()+1)
    for i in np.nonzero(~valid)[0]:
        yi,xi = np.nonzero(masks_padded==(i+1))
        ymed = np.median(yi)
        xmed = np.median(xi)
        imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
        centers[i,0] = yi[imin]
        centers[i,1] = xi[imin]        
    
    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]

    slices = scipy.ndimage.find_objects(masks)
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, 
                             n_iter=n_iter, device=device)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c

def masks_to_flows_cpu(masks, device=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    
    nmask = masks.max()
    slices = scipy.ndimage.find_objects(masks)
    # dia = utils.diameters(masks)[0]
    dia = diameters(masks)[0]
    s2 = (.15 * dia)**2
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y,x = np.nonzero(masks[sr, sc] == (i+1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
            xmed = x[imin]
            ymed = y[imin]

            d2 = (x-xmed)**2 + (y-ymed)**2
            mu_c[sr.start+y-1, sc.start+x-1] = np.exp(-d2/s2)

            niter = 2*np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly+2)*(lx+2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), niter)
            T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])

            dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx = T[y*lx + x+1] - T[y*lx + x-1]
            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((dy,dx))

    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    return mu, mu_c

def masks_to_flows(masks, use_gpu=False, device=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if TORCH_ENABLED and use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu 
    else:
        masks_to_flows_device = masks_to_flows_cpu
        
    if masks.ndim==3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], device=device)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], device=device)[0]
            mu[[0,1], :, :, x] += mu0
        return mu, None
    elif masks.ndim==2:
        mu = masks_to_flows_device(masks, device=device)[0]
        return mu, None
    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')



def labels_to_flows(labels, files=None):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell probability, flows[k][2] is Y flow, and flows[k][3] is X flow

    """

    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3:
        print('NOTE: computing flows for labels (could be done before to save time)')
        # compute flows        
        veci = [masks_to_flows(labels[n][0])[0] for n in trange(nimg)]
        # concatenate flows with cell probability
        flows = [np.concatenate((labels[n][[0]], labels[n][[0]]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        print('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


# #@njit(['(int16[:,:,:],float32[:], float32[:], float32[:,:])', '(float32[:,:,:],float32[:], float32[:], float32[:,:])'], cache=True)
# def map_coordinates(I, yc, xc, Y):
#     """
#     bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
#     Parameters
#     -------------
#     I : C x Ly x Lx
#     yc : ni
#         new y coordinates
#     xc : ni
#         new x coordinates
#     Y : C x ni
#         I sampled at (yc,xc)
#     """
#     C,Ly,Lx = I.shape
#     yc_floor = yc.astype(np.int32)
#     xc_floor = xc.astype(np.int32)
#     yc = yc - yc_floor
#     xc = xc - xc_floor
#     for i in range(yc_floor.shape[0]):
#         yf = min(Ly-1, max(0, yc_floor[i]))
#         xf = min(Lx-1, max(0, xc_floor[i]))
#         yf1= min(Ly-1, yf+1)
#         xf1= min(Lx-1, xf+1)
#         y = yc[i]
#         x = xc[i]
#         for c in range(C):
#             Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
#                       np.float32(I[c, yf, xf1]) * (1 - y) * x +
#                       np.float32(I[c, yf1, xf]) * y * (1 - x) +
#                       np.float32(I[c, yf1, xf1]) * y * x )

def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
    shape = dP.shape[1:]
    if use_gpu and TORCH_ENABLED:
        if device is None:
            device = torch_GPU
        pt = torch.from_numpy(p[[1,0]].T).double().to(device)
        pt = pt.unsqueeze(0).unsqueeze(0)
        pt[:,:,:,0] = (pt[:,:,:,0]/(shape[1]-1)) # normalize to between  0 and 1
        pt[:,:,:,1] = (pt[:,:,:,1]/(shape[0]-1)) # normalize to between  0 and 1
        pt = pt*2-1                       # normalize to between -1 and 1
        im = torch.from_numpy(dP[[1,0]]).double().to(device)
        im = im.unsqueeze(0)
        for k in range(2):
            im[:,k,:,:] /= (shape[1-k]-1) / 2.
        for t in range(niter):
            dPt = torch.nn.functional.grid_sample(im, pt)
            for k in range(2):
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] - dPt[:,k,:,:], -1., 1.)
        pt = (pt+1)*0.5
        pt[:,:,:,0] = pt[:,:,:,0] * (shape[1]-1)
        pt[:,:,:,1] = pt[:,:,:,1] * (shape[0]-1)
        return pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
    else:
        dPt = np.zeros(p.shape, np.float32)
        for t in range(niter):
            map_coordinates(dP, p[0], p[1], dPt)
            p[0] = np.minimum(shape[0]-1, np.maximum(0, p[0] - dPt[0]))
            p[1] = np.minimum(shape[1]-1, np.maximum(0, p[1] - dPt[1]))
        return p

# #@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
# def steps3D(p, dP, inds, niter):
#     """ run dynamics of pixels to recover masks in 3D
    
#     Euler integration of dynamics dP for niter steps

#     Parameters
#     ----------------

#     p: float32, 4D array
#         pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

#     dP: float32, 4D array
#         flows [axis x Lz x Ly x Lx]

#     inds: int32, 2D array
#         non-zero pixels to run dynamics on [npixels x 3]

#     niter: int32
#         number of iterations of dynamics to run

#     Returns
#     ---------------

#     p: float32, 4D array
#         final locations of each pixel after dynamics

#     """
#     shape = p.shape[1:]
#     for t in range(niter):
#         #pi = p.astype(np.int32)
#         for j in range(inds.shape[0]):
#             z = inds[j,0]
#             y = inds[j,1]
#             x = inds[j,2]
#             p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
#             p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] - dP[0,p0,p1,p2]))
#             p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] - dP[1,p0,p1,p2]))
#             p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] - dP[2,p0,p1,p2]))
#     return p

# #@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
# def steps2D(p, dP, inds, niter):
#     """ run dynamics of pixels to recover masks in 2D
    
#     Euler integration of dynamics dP for niter steps

#     Parameters
#     ----------------

#     p: float32, 3D array
#         pixel locations [axis x Ly x Lx] (start at initial meshgrid)

#     dP: float32, 3D array
#         flows [axis x Ly x Lx]

#     inds: int32, 2D array
#         non-zero pixels to run dynamics on [npixels x 2]

#     niter: int32
#         number of iterations of dynamics to run

#     Returns
#     ---------------

#     p: float32, 3D array
#         final locations of each pixel after dynamics

#     """
#     shape = p.shape[1:]
#     for t in range(niter):
#         #pi = p.astype(np.int32)
#         for j in range(inds.shape[0]):
#             y = inds[j,0]
#             x = inds[j,1]
#             p0, p1 = int(p[0,y,x]), int(p[1,y,x])
#             p[0,y,x] = min(shape[0]-1, max(0, p[0,y,x] - dP[0,p0,p1]))
#             p[1,y,x] = min(shape[1]-1, max(0, p[1,y,x] - dP[1,p0,p1]))
#     return p

def follow_flows(dP, niter=200, interp=True, use_gpu=True, device=None):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)
    if len(shape)>2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        #inds = np.array(np.nonzero(dP[0]!=0)).astype(np.int32).T
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        if inds.ndim < 2 or inds.shape[0] < 5:
            print('WARNING: no mask pixels found')
            return p
        if not interp:
            p = steps2D(p, dP, inds, niter)
        else:
            p[:,inds[:,0],inds[:,1]] = steps2D_interp(p[:,inds[:,0], inds[:,1]], 
                                                      dP, niter, use_gpu=use_gpu,
                                                      device=device)
    return p

def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    # merrors, _ = metrics.flow_error(masks, flows, use_gpu, device)
    merrors, _ = flow_error(masks, flows, use_gpu, device)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4, use_gpu=False, device=None):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 

    Parameters
    ----------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].

    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.

    rpad: int (optional, default 20)
        histogram edge padding

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)

    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.

    Returns
    ---------------

    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    _,counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0==i] = 0
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if M0.max()>0 and threshold is not None and threshold > 0 and flows is not None:
        M0 = remove_bad_flow_masks(M0, flows, threshold=threshold, use_gpu=use_gpu, device=device)
        _,M0 = np.unique(M0, return_inverse=True)
        M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0

# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### io.py #####################################################################
#################################################################################################################################

import os, datetime, gc, warnings, glob
from natsort import natsorted
import numpy as np
import cv2
import tifffile

# from . import utils, plot, transforms


try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

GUI = False
SERVER_UPLOAD = False


def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(filename)
        return img
    else:
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            print('ERROR: could not read file, %s'%e)
            return None

def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        tifffile.imsave(filename, arr)
    else:
        cv2.imwrite(filename, arr)

def get_image_files(folder, mask_filter, imf=None):
    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]
    image_names = []
    if imf is None:
        imf = ''
    image_names.extend(glob.glob(folder + '/*%s.png'%imf))
    image_names.extend(glob.glob(folder + '/*%s.jpg'%imf))
    image_names.extend(glob.glob(folder + '/*%s.jpeg'%imf))
    image_names.extend(glob.glob(folder + '/*%s.tif'%imf))
    image_names.extend(glob.glob(folder + '/*%s.tiff'%imf))
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) < len(mask_filter) 
                        for mask_filter in mask_filters])
        if len(imf)>0:
            igood &= imfile[-len(imf):]==imf
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names
        
def get_label_files(image_names, mask_filter, imf=None):
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0
        
    # check for flows
    if os.path.exists(label_names0[0] + '_flows.tif'):
        flow_names = [label_names0[n] + '_flows.tif' for n in range(nimg)]
    else:
        flow_names = [label_names[n] + '_flows.tif' for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        flow_names = None
    
    # check for masks
    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names, flow_names


def load_train_test_data(train_dir, test_dir=None, image_filter=None, mask_filter='_masks', unet=False):
    image_names = get_image_files(train_dir, mask_filter, imf=image_filter)
    nimg = len(image_names)
    images = [imread(image_names[n]) for n in range(nimg)]

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter, imf=image_filter)
    nimg = len(image_names)
    labels = [imread(label_names[n]) for n in range(nimg)]
    if flow_names is not None and not unet:
        for n in range(nimg):
            flows = imread(flow_names[n])
            if flows.shape[0]<4:
                labels[n] = np.concatenate((labels[n][np.newaxis,:,:], flows), axis=0) 
            else:
                labels[n] = flows
            
    # testing data
    test_images, test_labels, image_names_test = None, None, None
    if test_dir is not None:
        image_names_test = get_image_files(test_dir, mask_filter, imf=image_filter)
        label_names_test, flow_names_test = get_label_files(image_names_test, mask_filter, imf=image_filter)
        nimg = len(image_names_test)
        test_images = [imread(image_names_test[n]) for n in range(nimg)]
        test_labels = [imread(label_names_test[n]) for n in range(nimg)]
        if flow_names_test is not None and not unet:
            for n in range(nimg):
                flows = imread(flow_names_test[n])
                if flows.shape[0]<4:
                    test_labels[n] = np.concatenate((test_labels[n][np.newaxis,:,:], flows), axis=0) 
                else:
                    test_labels[n] = flows
    return images, labels, image_names, test_images, test_labels, image_names_test



def masks_flows_to_seg(images, masks, flows, diams, file_names, channels=None):
    """ save output of model eval to be loaded in GUI 

    can be list output (run on multiple images) or single output (run on single image)

    saved to file_names[k]+'_seg.npy'
    
    Parameters
    -------------

    images: (list of) 2D or 3D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from Cellpose.eval

    diams: float array
        diameters used to run Cellpose

    file_names: (list of) str
        names of files of images

    channels: list of int (optional, default None)
        channels used to run Cellpose    
    
    """
    
    if channels is None:
        channels = [0,0]
    
    if isinstance(masks, list):
        for k, [image, mask, flow, diam, file_name] in enumerate(zip(images, masks, flows, diams, file_names)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, diam, file_name, channels_img)
        return

    if len(channels)==1:
        channels = channels[0]

    flowi = []
    if flows[0].ndim==3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...])
    else:
        flowi.append(flows[0])
    if flows[0].ndim==3:
        # cellprob = (np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8)
        cellprob = (np.clip(normalize99(flows[2]),0,1) * 255).astype(np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis,...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis,...]
    else:
        # flowi.append((np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8))
        flowi.append((np.clip(normalize99(flows[2]),0,1) * 255).astype(np.uint8))
        flowi.append((flows[1][0]/10 * 127 + 127).astype(np.uint8))
    if len(flows)>2:
        flowi.append(flows[3])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis,...]), axis=0))
    # outlines = masks * utils.masks_to_outlines(masks)
    outlines = masks * masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]
    if masks.ndim==3:
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                        'masks': masks.astype(np.uint16) if outlines.max()<2**16-1 else masks.astype(np.uint32),
                        'chan_choose': channels,
                        'img': images,
                        'ismanual': np.zeros(masks.max(), np.bool),
                        'filename': file_names,
                        'flows': flowi,
                        'est_diam': diams})
    else:
        if images.shape[0]<8:
            np.transpose(images, (1,2,0))
        np.save(base+ '_seg.npy',
                    {'img': images,
                        'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                     'masks': masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32),
                     'chan_choose': channels,
                     'ismanual': np.zeros(masks.max(), np.bool),
                     'filename': file_names,
                     'flows': flowi,
                     'est_diam': diams})    

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True) 
    
        does not work for 3D images
    
    """
    save_masks(images, masks, flows, file_names, png=True)

def save_masks(images, masks, flows, file_names, png=True, tif=False):
    """ save masks + nicely plotted segmentation image to png and/or tiff

    if png, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.png'

    if tif, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.tif'

    if png and matplotlib installed, full segmentation figure is saved to file_names[k]+'_cp.png'

    only tif option works for 3D data
    
    Parameters
    -------------

    images: (list of) 2D, 3D or 4D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from Cellpose.eval

    file_names: (list of) str
        names of files of images
    
    """
    
    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif)
        return
    
    if masks.ndim > 2 and not tif:
        raise ValueError('cannot save 3D outputs as PNG, use tif option instead')
    base = os.path.splitext(file_names)[0]
    exts = []
    if masks.ndim > 2 or masks.max()>2**16-1:
        png = False
        tif = True
    if png:    
        exts.append('.png')
    if tif:
        exts.append('.tif')

    # convert to uint16 if possible so can save as PNG if needed
    masks = masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32)
    
    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            imsave(base + '_cp_masks' + ext, masks)

    if png and MATPLOTLIB and not min(images.shape) > 3:
        img = images.copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        elif img.shape[0]<8:
            np.transpose(img, (1,2,0))
        
        fig = plt.figure(figsize=(12,3))
        # can save images (set save_dir=None if not)
        # plot.show_segmentation(fig, img, masks, flows[0])
        show_segmentation(fig, img, masks, flows[0])
        fig.savefig(base + '_cp_output.png', dpi=300)
        plt.close(fig)

    if masks.ndim < 3: 
        # outlines = utils.outlines_list(masks)
        outlines = outlines_list(masks)
        outlines_to_text(base, outlines)

            

def _initialize_images(parent, image, resize, X2):
    """ format image for GUI """
    parent.onechan=False
    if image.ndim > 3:
        # make tiff Z x channels x W x H
        if image.shape[0]<4:
            # tiff is channels x Z x W x H
            image = np.transpose(image, (1,0,2,3))
        elif image.shape[-1]<4:
            # tiff is Z x W x H x channels
            image = np.transpose(image, (0,3,1,2))
        # fill in with blank channels to make 3 channels
        if image.shape[1] < 3:
            shape = image.shape
            image = np.concatenate((image,
                            np.zeros((shape[0], 3-shape[1], shape[2], shape[3]), dtype=np.uint8)), axis=1)
            if 3-shape[1]>1:
                parent.onechan=True
        image = np.transpose(image, (0,2,3,1))
    elif image.ndim==3:
        if image.shape[0] < 5:
            image = np.transpose(image, (1,2,0))

        if image.shape[-1] < 3:
            shape = image.shape
            image = np.concatenate((image,
                                       np.zeros((shape[0], shape[1], 3-shape[2]),
                                        dtype=type(image[0,0,0]))), axis=-1)
            if 3-shape[2]>1:
                parent.onechan=True
            image = image[np.newaxis,...]
        elif image.shape[-1]<5 and image.shape[-1]>2:
            image = image[:,:,:3]
            image = image[np.newaxis,...]
    else:
        image = image[np.newaxis,...]

    parent.stack = image
    parent.NZ = len(parent.stack)
    parent.scroll.setMaximum(parent.NZ-1)
    if parent.stack.max()>255 or parent.stack.min()<0.0 or parent.stack.max()<=50.0:
        parent.stack = parent.stack.astype(np.float32)
        parent.stack -= parent.stack.min()
        parent.stack /= parent.stack.max()
        parent.stack *= 255
    del image
    gc.collect()

    parent.stack = list(parent.stack)
    for k,img in enumerate(parent.stack):
        # if grayscale make 3D
        if resize != -1:
            # img = transforms._image_resizer(img, resize=resize, to_uint8=False)
            img = _image_resizer(img, resize=resize, to_uint8=False)
        if img.ndim==2:
            img = np.tile(img[:,:,np.newaxis], (1,1,3))
            parent.onechan=True
        if X2!=0:
            # img = transforms._X2zoom(img, X2=X2)
            img = _X2zoom(img, X2=X2)
        parent.stack[k] = img

    parent.imask=0
    print(parent.NZ, parent.stack[0].shape)
    parent.Ly, parent.Lx = img.shape[0], img.shape[1]
    parent.stack = np.array(parent.stack)
    parent.layers = 0*np.ones((parent.NZ,parent.Ly,parent.Lx,4), np.uint8)
    if parent.autobtn.isChecked() or len(parent.saturation)!=parent.NZ:
        parent.compute_saturation()
    parent.compute_scale()
    parent.currentZ = int(np.floor(parent.NZ/2))
    parent.scroll.setValue(parent.currentZ)
    parent.zpos.setText(str(parent.currentZ))

def _load_seg(parent, filename=None, image=None, image_file=None):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    try:
        dat = np.load(filename, allow_pickle=True).item()
        dat['outlines']
        parent.loaded = True
    except:
        parent.loaded = False
        print('not NPY')
        return

    parent.reset()
    if image is None:
        found_image = False
        if 'filename' in dat:
            parent.filename = dat['filename']
            if os.path.isfile(parent.filename):
                parent.filename = dat['filename']
                found_image = True
            else:
                imgname = os.path.split(parent.filename)[1]
                root = os.path.split(filename)[0]
                parent.filename = root+'/'+imgname
                if os.path.isfile(parent.filename):
                    found_image = True
        if found_image:
            try:
                image = imread(parent.filename)
            except:
                parent.loaded = False
                found_image = False
                print('ERROR: cannot find image file, loading from npy')
        if not found_image:
            parent.filename = filename[:-11]
            if 'img' in dat:
                image = dat['img']
            else:
                print('ERROR: no image file found and no image in npy')
                return
    else:
        parent.filename = image_file
    print(parent.filename)

    if 'X2' in dat:
        parent.X2 = dat['X2']
    else:
        parent.X2 = 0
    if 'resize' in dat:
        parent.resize = dat['resize']
    elif 'img' in dat:
        if max(image.shape) > max(dat['img'].shape):
            parent.resize = max(dat['img'].shape)
    else:
        parent.resize = -1
    _initialize_images(parent, image, resize=parent.resize, X2=parent.X2)
    if 'chan_choose' in dat:
        parent.ChannelChoose[0].setCurrentIndex(dat['chan_choose'][0])
        parent.ChannelChoose[1].setCurrentIndex(dat['chan_choose'][1])
    if 'outlines' in dat:
        if isinstance(dat['outlines'], list):
            # old way of saving files
            dat['outlines'] = dat['outlines'][::-1]
            for k, outline in enumerate(dat['outlines']):
                if 'colors' in dat:
                    color = dat['colors'][k]
                else:
                    col_rand = np.random.randint(1000)
                    color = parent.colormap[col_rand,:3]
                median = parent.add_mask(points=outline, color=color)
                if median is not None:
                    parent.cellcolors.append(color)
                    parent.ncells+=1
        else:
            if dat['masks'].ndim==2:
                dat['masks'] = dat['masks'][np.newaxis,:,:]
                dat['outlines'] = dat['outlines'][np.newaxis,:,:]
            if dat['masks'].min()==-1:
                dat['masks'] += 1
                dat['outlines'] += 1
            if 'colors' in dat:
                colors = dat['colors']
            else:
                col_rand = np.random.randint(0, 1000, (dat['masks'].max(),))
                colors = parent.colormap[col_rand,:3]
            parent.cellpix = dat['masks']
            parent.outpix = dat['outlines']
            parent.cellcolors.extend(colors)
            parent.ncells = parent.cellpix.max()
            parent.draw_masks()
            if 'est_diam' in dat:
                parent.Diameter.setText('%0.1f'%dat['est_diam'])
                parent.diameter = dat['est_diam']
                parent.compute_scale()

            if parent.masksOn or parent.outlinesOn and not (parent.masksOn and parent.outlinesOn):
                parent.redraw_masks(masks=parent.masksOn, outlines=parent.outlinesOn)
        if 'zdraw' in dat:
            parent.zdraw = dat['zdraw']
        else:
            parent.zdraw = [None for n in range(parent.ncells)]
        parent.loaded = True
        print('%d masks found'%(parent.ncells))
    else:
        parent.clear_all()

    parent.ismanual = np.zeros(parent.ncells, np.bool)
    if 'ismanual' in dat:
        if len(dat['ismanual']) == parent.ncells:
            parent.ismanual = dat['ismanual']

    if 'current_channel' in dat:
        parent.color = (dat['current_channel']+2)%5
        parent.RGBDropDown.setCurrentIndex(parent.color)

    if 'flows' in dat:
        parent.flows = dat['flows']
        if parent.flows[0].shape[-3]!=dat['masks'].shape[-2]:
            Ly, Lx = dat['masks'].shape[-2:]
            parent.flows[0] = cv2.resize(parent.flows[0][0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
            parent.flows[1] = cv2.resize(parent.flows[1][0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
        try:
            if parent.NZ==1:
                parent.threshslider.setEnabled(True)
                parent.probslider.setEnabled(True)
            else:
                parent.threshslider.setEnabled(False)
                parent.probslider.setEnabled(False)
        except:
            try:
                if len(parent.flows[0])>0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[],[],[],[],[[]]]
            parent.threshslider.setEnabled(False)
            parent.probslider.setEnabled(False)
            
    parent.enable_buttons()
    del dat
    gc.collect()

def _load_masks(parent, filename=None):
    """ load zeros-based masks (0=no cell, 1=cell 1, ...) """
    masks = imread(filename)
    outlines = None
    if masks.ndim>3:
        # Z x nchannels x Ly x Lx
        if masks.shape[-1]>5:
            parent.flows = list(np.transpose(masks[:,:,:,2:], (3,0,1,2)))
            outlines = masks[...,1]
            masks = masks[...,0]
        else:
            parent.flows = list(np.transpose(masks[:,:,:,1:], (3,0,1,2)))
            masks = masks[...,0]
    elif masks.ndim==3:
        if masks.shape[-1]<5:
            masks = masks[np.newaxis,:,:,0]
    elif masks.ndim<3:
        masks = masks[np.newaxis,:,:]
    # masks should be Z x Ly x Lx
    if masks.shape[0]!=parent.NZ:
        print('ERROR: masks are not same depth (number of planes) as image stack')
        return
    print('%d masks found'%(len(np.unique(masks))-1))

    _masks_to_gui(parent, masks, outlines)

    parent.update_plot()

def _masks_to_gui(parent, masks, outlines=None):
    """ masks loaded into GUI """
    # get unique values
    shape = masks.shape
    _, masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape)
    masks = masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32)
    parent.cellpix = masks

    # get outlines
    if outlines is None:
        parent.outpix = np.zeros_like(masks)
        for z in range(parent.NZ):
            # outlines = utils.masks_to_outlines(masks[z])
            outlines = masks_to_outlines(masks[z])
            parent.outpix[z] = outlines * masks[z]
            if z%50==0:
                print('plane %d outlines processed'%z)
    else:
        parent.outpix = outlines
        shape = parent.outpix.shape
        _,parent.outpix = np.unique(parent.outpix, return_inverse=True)
        parent.outpix = np.reshape(parent.outpix, shape)

    parent.ncells = parent.cellpix.max()
    colors = parent.colormap[np.random.randint(0,1000,size=parent.ncells), :3]
    parent.cellcolors = list(np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8))
    parent.draw_masks()
    if parent.ncells>0:
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, np.bool)
    parent.zdraw = list(-1*np.ones(parent.ncells, np.int16))
    parent.update_plot()

def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('saving 2D masks to png')
        imsave(base + '_cp_masks.png', parent.cellpix[0])
    else:
        print('saving 3D masks to tiff')
        imsave(base + '_cp_masks.tif', parent.cellpix)

def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        print('saving 2D outlines to text file, see docs for info to load into ImageJ')    
        # outlines = utils.outlines_list(parent.cellpix[0])
        outlines = outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        print('ERROR: cannot save 3D outlines')
    

def _save_sets(parent):
    """ save masks to *_seg.npy """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ > 1 and parent.is_stack:
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix,
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix,
                 'current_channel': (parent.color-2)%5,
                 'filename': parent.filename,
                 'flows': parent.flows,
                 'zdraw': parent.zdraw})
    else:
        image = parent.chanchoose(parent.stack[parent.currentZ].copy())
        if image.ndim < 4:
            image = image[np.newaxis,...]
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix.squeeze(),
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix.squeeze(),
                 'chan_choose': [parent.ChannelChoose[0].currentIndex(),
                                 parent.ChannelChoose[1].currentIndex()],
                 'img': image.squeeze(),
                 'ismanual': parent.ismanual,
                 'X2': parent.X2,
                 'filename': parent.filename,
                 'flows': parent.flows})
    #print(parent.point_sets)
    print('--- %d ROIs saved chan1 %s, chan2 %s'%(parent.ncells,
                                                  parent.ChannelChoose[0].currentText(),
                                                  parent.ChannelChoose[1].currentText()))

# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### metrics.py ################################################################
#################################################################################################################################

import numpy as np
# from . import utils, dynamics
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve, mean

from njitFunc import _label_overlap

def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds

def boundary_scores(masks_true, masks_pred, scales):
    """ boundary precision / recall / Fscore """
    # diams = [utils.diameters(lbl)[0] for lbl in masks_true]
    diams = [diameters(lbl)[0] for lbl in masks_true]
    precision = np.zeros((len(scales), len(masks_true)))
    recall = np.zeros((len(scales), len(masks_true)))
    fscore = np.zeros((len(scales), len(masks_true)))
    for j, scale in enumerate(scales):
        for n in range(len(masks_true)):
            diam = max(1, scale * diams[n])
            # rs, ys, xs = utils.circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
            rs, ys, xs = circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
            filt = (rs <= diam).astype(np.float32)
            # otrue = utils.masks_to_outlines(masks_true[n])
            otrue = masks_to_outlines(masks_true[n])
            otrue = convolve(otrue, filt)
            # opred = utils.masks_to_outlines(masks_pred[n])
            opred = masks_to_outlines(masks_pred[n])
            opred = convolve(opred, filt)
            tp = np.logical_and(otrue==1, opred==1).sum()
            fp = np.logical_and(otrue==0, opred==1).sum()
            fn = np.logical_and(otrue==1, opred==0).sum()
            precision[j,n] = tp / (tp + fp)
            recall[j,n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return precision, recall, fscore


def aggregated_jaccard_index(masks_true, masks_pred):
    """ AJI = intersection of all matched masks / union of all masks 
    
    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    aji : aggregated jaccard index for each set of masks

    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n]>0, masks_pred[n]>0).sum()
        overlap = overlap[inds[preds>0]+1, preds[preds>0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji 


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

# #@jit(nopython=True)
# def _label_overlap(x, y):
#     """ fast function to get pixel overlaps between masks in x and y 
    
#     Parameters
#     ------------

#     x: ND-array, int
#         where 0=NO masks; 1,2... are mask labels
#     y: ND-array, int
#         where 0=NO masks; 1,2... are mask labels

#     Returns
#     ------------

#     overlap: ND-array, int
#         matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
#     """
#     x = x.ravel()
#     y = y.ravel()
#     overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
#     for i in range(len(x)):
#         overlap[x[i],y[i]] += 1
#     return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def flow_error(maski, dP_net, use_gpu=False, device=None):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------
    
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return

    # ensure unique masks
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
  
    # flows predicted from estimated masks
    # dP_masks = dynamics.masks_to_flows(maski, use_gpu=use_gpu, device=device)[0]
    dP_masks = masks_to_flows(maski, use_gpu=use_gpu, device=device)[0]
    
    # difference between predicted flows vs mask flows
    flow_errors=np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski,
                            index=np.arange(1, maski.max()+1))
    
    return flow_errors, dP_masks


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### models.py #################################################################
#################################################################################################################################

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile
from urllib.request import urlopen
from scipy.ndimage import median_filter
import cv2

# from . import transforms, dynamics, utils, plot, metrics, core
# from .core import UnetModel, assign_device, check_mkl, use_gpu, convert_images, MXNET_ENABLED, parse_model_string

urls = ['https://www.cellpose.org/models/cyto_0',
        # 'https://www.cellpose.org/models/cyto_1',
        # 'https://www.cellpose.org/models/cyto_2',
        # 'https://www.cellpose.org/models/cyto_3',
        'https://www.cellpose.org/models/size_cyto_0.npy',
        'https://www.cellpose.org/models/cytotorch_0',
        # 'https://www.cellpose.org/models/cytotorch_1',
        # 'https://www.cellpose.org/models/cytotorch_2',
        # 'https://www.cellpose.org/models/cytotorch_3',
        'https://www.cellpose.org/models/size_cytotorch_0.npy']


def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
            Thanks to torch, slightly modified
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def download_model_weights(urls=urls):
    # cellpose directory
    #cp_dir = pathlib.Path.home().joinpath('.cellpose')
    #cp_dir.mkdir(exist_ok=True)
    curr_dir = pathlib.Path.cwd()
    model_dir = curr_dir.joinpath('model_weights')
    model_dir.mkdir(exist_ok=True)

    for url in urls:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            # utils.download_url_to_file(url, cached_file, progress=True)
            download_url_to_file(url, cached_file, progress=True)

download_model_weights()
model_dir = pathlib.Path.cwd().joinpath('model_weights')

def dx_to_circ(dP):
    """ dP is 2 x Y x X => 'optic' flow representation """
    sc = max(np.percentile(dP[0], 99), np.percentile(dP[0], 1))
    Y = np.clip(dP[0] / sc, -1, 1)
    sc = max(np.percentile(dP[1], 99), np.percentile(dP[1], 1))
    X = np.clip(dP[1] / sc, -1, 1)
    H = (np.arctan2(Y, X) + np.pi) / (2*np.pi) * 179
    # S = np.clip(utils.normalize99(dP[0]**2 + dP[1]**2), 0.0, 1.0) * 255
    S = np.clip(normalize99(dP[0]**2 + dP[1]**2), 0.0, 1.0) * 255
    V = np.ones_like(S) * 255
    HSV = np.stack((H,S,S), axis=-1)
    flow = cv2.cvtColor(HSV.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return flow

class Pose():
    """ main model which combines SizeModel and CellposeModel

    Parameters
    ----------

    gpu: bool (optional, default False)
        whether or not to use GPU,will check if GPU available

    model_type: str (optional, default 'cyto')
        'cyto'=cytoplasm model; 'nuclei'=nucleus model

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    device: gpu device (optional, default None)
        where model is saved (e.g. mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4) or torch.cuda.device(4))

    torch: bool (optional, default False)
        run model using torch if available

    """
    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, torch=True, Upscale_Factor=4):
        super(Pose, self).__init__()

        self.Upscale_Factor = Upscale_Factor

        if not torch:
            if not MXNET_ENABLED:
                print('WARNING: mxnet not installed, using torch')
                torch = True
        self.torch = torch
        torch_str = ['','torch'][self.torch]
        
        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        model_type = 'cyto' if model_type is None else model_type
        self.pretrained_model = [os.fspath(model_dir.joinpath('%s%s_%d'%(model_type,torch_str,j))) for j in range(4)]
        self.pretrained_size = os.fspath(model_dir.joinpath('size_%s%s_0.npy'%(model_type,torch_str)))
        self.diam_mean = 30. if model_type=='cyto' else 17.
        
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]

        self.cp = PoseModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean, torch=self.torch, Upscale_Factor=Upscale_Factor)
        self.cp.model_type = model_type

        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True, diameter=30., do_3D=False, anisotropy=None,
             net_avg=True, augment=False, tile=True, tile_overlap=0.1, resample=False, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, 
              stitch_threshold=0.0, rescale=None, progress=None):
        """ run cellpose and get masks

        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

        batch_size: int (optional, default 8)
            number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)

        channels: list (optional, default None)
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
            For instance, to segment grayscale images, input [0,0]. To segment images with cells
            in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
            image with cells in green and nuclei in blue, input [[0,0], [2,3]].

        invert: bool (optional, default False)
            invert image pixel intensity before running network (if True, image is also normalized)

        normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        diameter: float (optional, default 30.)
            if set to None, then diameter is automatically estimated if size model is loaded

        do_3D: bool (optional, default False)
            set to True to run 3D segmentation on 4D image input

        anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        resample: bool (optional, default False)
            run dynamics at original image size (will be slower but create more accurate boundaries)

        interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

        cellprob_threshold: float (optional, default 0.0)
            cell probability threshold (all pixels with prob above threshold kept for masks)

        min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D and equal image sizes, masks are stitched in 3D to return volume segmentation

        rescale: float (optional, default None)
            if diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI

        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = flows at each pixel
            flows[k][2] = the cell probability centered at 0.0

        styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image

        diams: list of diameters, or float (if do_3D=True)

        """

        if not isinstance(x,list):
            nolist = True
            if x.ndim < 2 or x.ndim > 5:
                raise ValueError('%dD images not supported'%x.ndim)
            if x.ndim==4:
                if do_3D:
                    x = [x]
                else:
                    x = list(x)
                    nolist = False
            elif x.ndim==5: 
                if do_3D:
                    x = list(x)
                    nolist = False
                else:
                    raise ValueError('4D images must be processed using 3D')
            else:
                x = [x]
        else:
            nolist = False
            for xi in x:
                if xi.ndim < 2 or xi.ndim > 5:
                    raise ValueError('%dD images not supported'%xi.ndim)
            
        tic0 = time.time()

        nimg = len(x)
        print('processing %d image(s)'%nimg)
        # make rescale into length of x
        if diameter is not None and not (not isinstance(diameter, (list, np.ndarray)) and 
                (diameter==0 or (diameter==30. and rescale is not None))):    
            if not isinstance(diameter, (list, np.ndarray)) or len(diameter)==1 or len(diameter)<nimg:
                diams = diameter * np.ones(nimg, np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / diams
        else:
            if rescale is not None and (not isinstance(rescale, (list, np.ndarray)) or len(rescale)==1):
                rescale = rescale * np.ones(nimg, np.float32)
            if self.pretrained_size is not None and rescale is None and not do_3D:
                tic = time.time()
                diams, _ = self.sz.eval(x, channels=channels, invert=invert, batch_size=batch_size, 
                                        augment=augment, tile=tile)
                rescale = self.diam_mean / diams
                print('estimated cell diameters for %d image(s) in %0.2f sec'%(nimg, time.time()-tic))
                print('>>> diameter(s) = ', diams)
            else:
                if rescale is None:
                    if do_3D:
                        rescale = np.ones(1)
                    else:
                        rescale = np.ones(nimg, np.float32)
                diams = self.diam_mean / rescale
        tic = time.time()
        masks, flows, styles = self.cp.eval(x, 
                                            batch_size=batch_size, 
                                            invert=invert, 
                                            rescale=rescale, 
                                            anisotropy=anisotropy, 
                                            channels=channels, 
                                            augment=augment, 
                                            tile=tile, 
                                            do_3D=do_3D, 
                                            net_avg=net_avg, progress=progress,
                                            tile_overlap=tile_overlap,
                                            resample=resample,
                                            interp=interp,
                                            flow_threshold=flow_threshold, 
                                            cellprob_threshold=cellprob_threshold,
                                            min_size=min_size, 
                                            stitch_threshold=stitch_threshold)
        print('estimated masks for %d image(s) in %0.2f sec'%(nimg, time.time()-tic))
        print('>>>> TOTAL TIME %0.2f sec'%(time.time()-tic0))
        
        if nolist:
            masks, flows, styles, diams = masks[0], flows[0], styles[0], diams[0]
        
        return masks, flows, styles, diams

class PoseModel(UnetModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available

    pretrained_model: str or list of strings (optional, default False)
        path to pretrained cellpose model(s), if False, no model loaded;
        if None, built-in 'cyto' model loaded

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model

    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))

    """

    def __init__(self, gpu=False, pretrained_model=False, torch=True,
                    diam_mean=30., net_avg=True, device=None,
                    residual_on=True, style_on=True, concatenation=False, Upscale_Factor=4):

        self.Upscale_Factor = Upscale_Factor

        if not torch:
            if not MXNET_ENABLED:
                print('WARNING: mxnet not installed, using torch')
                torch = True
        self.torch = torch
        
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        nclasses = 3 # 3 prediction maps (dY, dX and cellprob)
        self.nclasses = nclasses 
        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        # load default cyto model if pretrained_model is None
        elif pretrained_model is None:
            torch_str = ['','torch'][self.torch]
            pretrained_model = [os.fspath(model_dir.joinpath('cyto%s_%d'%(torch_str,j))) for j in range(4)] if net_avg else os.fspath(model_dir.joinpath('cyto_0'))
            self.diam_mean = 30.
            residual_on, style_on, concatenation = True, True, False
        
        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         nclasses=nclasses, torch=torch)
        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_model(self.pretrained_model, cpu=(not self.gpu))
        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])
                                                                                
    def eval(self, imgs, batch_size=8, channels=None, normalize=True, invert=False, 
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=True, 
             augment=False, tile=True, tile_overlap=0.1,
             resample=False, interp=True, flow_threshold=0.4, cellprob_threshold=0.0, compute_masks=True, 
             min_size=15, stitch_threshold=0.0, return_conv=False, progress=None):
        """
            segment list of images imgs, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            imgs: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D images

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            diameter: float (optional, default None)
                diameter for each image (only used if rescale is None), 
                if diameter is None, set to diam_mean

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default False)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            cellprob_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            return_conv: bool (optional, default False)
                return activations from final convolutional layer

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        x, nolist = convert_images(imgs.copy(), channels, do_3D, normalize, invert)
        
        nimg = len(x)
        self.batch_size = batch_size
        
        styles = []
        flows = []
        masks = []

        if rescale is None:
            if diameter is not None:
                if not isinstance(diameter, (list, np.ndarray)):
                    diameter = diameter * np.ones(nimg)
                rescale = self.diam_mean / diameter
            else:
                rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)

        iterator = trange(nimg) if nimg>1 else range(nimg)
        
        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
            if not self.torch:
                self.net.collect_params().grad_req = 'null'

        if not do_3D:
            flow_time = 0
            net_time = 0
            for i in iterator:
                img = x[i].copy()
                Ly,Lx = img.shape[:2]

                tic = time.time()
                shape = img.shape
                # rescale image for flow computation
                # img = transforms.resize_image(img, rsz=rescale[i])
                img = resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg, 
                                          augment=augment, tile=tile,
                                          tile_overlap=tile_overlap,
                                          return_conv=return_conv)
                net_time += time.time() - tic
                if progress is not None:
                    progress.setValue(55)
                styles.append(style)
                # if True:#resample:
                #     # y = transforms.resize_image(y, shape[-3], shape[-2])
                #     y = resize_image(y, shape[-3]*self.Upscale_Factor, shape[-2]*self.Upscale_Factor)
                cellprob = y[:,:,-1]
                dP = y[:,:,:2].transpose((2,0,1))
                if compute_masks:
                    niter = 1 / rescale[i] * 200
                    if progress is not None:
                        progress.setValue(65)
                    tic=time.time()
                    # p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., 
                    p = follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., 
                                                   niter=niter, interp=interp, use_gpu=self.gpu,
                                                   device=self.device)
                    # maski = dynamics.get_masks(p, iscell=(cellprob>cellprob_threshold),
                    maski = get_masks(p, iscell=(cellprob>cellprob_threshold),
                                                    flows=dP, threshold=flow_threshold,
                                                    use_gpu=self.gpu, device=self.device)
                    # maski = utils.fill_holes_and_remove_small_masks(maski)
                    maski = fill_holes_and_remove_small_masks(maski)
                    # maski = transforms.resize_image(maski, shape[-3], shape[-2], interpolation=cv2.INTER_NEAREST)
                    maski = resize_image(maski, shape[-3]*self.Upscale_Factor, shape[-2]*self.Upscale_Factor, interpolation=cv2.INTER_NEAREST)
                    if progress is not None:
                        progress.setValue(75)
                    #dP = np.concatenate((dP, np.zeros((1,dP.shape[1],dP.shape[2]), np.uint8)), axis=0)
                else:
                    p = []
                    maski = []
                flows.append([dx_to_circ(dP), dP, cellprob, p])
                
                if return_conv:
                    flows[-1].append(y[:,:,3:])
                masks.append(maski)
                
                flow_time += time.time() - tic
            if compute_masks:
                print('time spent: running network %0.2fs; flow+mask computation %0.2f'%(net_time, flow_time))

            if stitch_threshold > 0.0 and nimg > 1 and all([m.shape==masks[0].shape for m in masks]):
                print('stitching %d masks using stitch_threshold=%0.3f to make 3D masks'%(nimg, stitch_threshold))
                # masks = utils.stitch3D(np.array(masks), stitch_threshold=stitch_threshold)
                masks = stitch3D(np.array(masks), stitch_threshold=stitch_threshold)
        else:
            for i in iterator:
                tic=time.time()
                shape = x[i].shape
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile, 
                                         tile_overlap=tile_overlap, progress=progress)
                cellprob = yf[0][2] + yf[1][2] + yf[2][2]
                dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), 
                                axis=0) # (dZ, dY, dX)
                print('flows computed %2.2fs'%(time.time()-tic))
                # ** mask out values using cellprob to increase speed and reduce memory requirements **
                # yout = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.)
                yout = follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.)
                print('dynamics computed %2.2fs'%(time.time()-tic))
                # maski = dynamics.get_masks(yout, iscell=(cellprob>cellprob_threshold))
                maski = get_masks(yout, iscell=(cellprob>cellprob_threshold))
                # maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                maski = fill_holes_and_remove_small_masks(maski, min_size=min_size)
                print('masks computed %2.2fs'%(time.time()-tic))
                flow = np.array([dx_to_circ(dP[1:,i]) for i in range(dP.shape[1])])
                flows.append([flow, dP, cellprob, yout])
                masks.append(maski)
                styles.append(style)
        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]
        return masks, flows, styles

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        
        veci = 5. * self._to_device(lbl[:,1:])
        lbl  = self._to_device(lbl[:,0]>.5)
        loss = self.criterion(y[:,:2] , veci) 
        if self.torch:
            loss /= 2.
        loss2 = self.criterion2(y[:,2] , lbl)
        loss = loss + loss2
        return loss


    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None, 
              save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, rescale=True):

        """ train network with images train_data 
        
            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels; 
                can include flows as additional images
        
            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            pretrained_model: string (default, None)
                path to pretrained_model to start from, if None it is trained from scratch

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float (default, 0.2)
                learning rate for training

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training, 
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)

        """

        # train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize)
        train_data, train_labels, test_data, test_labels, run_test = reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize)

        # check if train_labels have flows
        # train_flows = dynamics.labels_to_flows(train_labels, files=train_files)
        train_flows = labels_to_flows(train_labels, files=train_files)
        if run_test:
            # test_flows = dynamics.labels_to_flows(test_labels, files=test_files)
            test_flows = labels_to_flows(test_labels, files=test_files)
        else:
            test_flows = None
        
        model_path = self._train_net(train_data, train_flows, 
                                     test_data, test_flows,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, momentum, weight_decay, batch_size, rescale)
        self.pretrained_model = model_path
        return model_path

class SizeModel():
    """ linear regression model for determining the size of objects in image
        used to rescale before input to cp_model
        uses styles from cp_model

        Parameters
        -------------------

        cp_model: UnetModel or CellposeModel
            model from which to get styles

        device: mxnet device (optional, default mx.cpu())
            where cellpose model is saved (mx.gpu() or mx.cpu())

        pretrained_size: str
            path to pretrained size model

    """
    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)

        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean
        self.torch = self.cp.torch
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params['diam_mean']
        if not hasattr(self.cp, 'pretrained_model'):
            raise ValueError('provided model does not have a pretrained_model')
        
    def eval(self, imgs=None, styles=None, channels=None, normalize=True, invert=False, augment=False, tile=True,
                batch_size=8, progress=None):
        """ use images imgs to produce style or use style input to predict size of objects in image

            Object size estimation is done in two steps:
            1. use a linear regression model to predict size from style in image
            2. resize image to predicted size and run CellposeModel to get output masks.
                Take the median object size of the predicted masks as the final predicted size.

            Parameters
            -------------------

            imgs: list or array of images (optional, default None)
                can be list of 2D/3D images, or array of 2D/3D images

            styles: list or array of styles (optional, default None)
                styles for images x - if x is None then styles must not be None

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            diam: array, float
                final estimated diameters from images x or styles style after running both steps

            diam_style: array, float
                estimated diameters from style alone

        """
        if styles is None and imgs is None:
            raise ValueError('no image or features given')
        
        if progress is not None:
            progress.setValue(10)
        
        if imgs is not None:
            x, nolist = convert_images(imgs.copy(), channels, False, normalize, invert)
            nimg = len(x)
        
        if styles is None:
            print('computing styles from images')
            styles = self.cp.eval(x, net_avg=False, augment=augment, tile=tile, compute_masks=False)[-1]
            if progress is not None:
                progress.setValue(30)
            diam_style = self._size_estimation(np.array(styles))
            if progress is not None:
                progress.setValue(50)
        else:
            styles = np.array(styles) if isinstance(styles, list) else styles
            diam_style = self._size_estimation(styles)
        diam_style[np.isnan(diam_style)] = self.diam_mean

        if imgs is not None:
            masks = self.cp.eval(x, rescale=self.diam_mean/diam_style, net_avg=False, 
                                augment=augment, tile=tile, interp=False)[0]
            # diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
            diam = np.array([diameters(masks[i])[0] for i in range(nimg)])
            if progress is not None:
                progress.setValue(100)
            if hasattr(self, 'model_type') and (self.model_type=='nuclei' or self.model_type=='cyto') and not self.torch:
                diam_style /= (np.pi**0.5)/2
                diam[diam==0] = self.diam_mean / ((np.pi**0.5)/2)
                diam[np.isnan(diam)] = self.diam_mean / ((np.pi**0.5)/2)
            else:
                diam[diam==0] = self.diam_mean
                diam[np.isnan(diam)] = self.diam_mean
        else:
            diam = diam_style
            print('no images provided, using diameters estimated from styles alone')
        if nolist:
            return diam[0], diam_style[0]
        else:
            return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size 
        
            sizes were estimated using "diameters" from square estimates not circles; 
            therefore a conversion factor is included (to be removed)
        
        """
        szest = np.exp(self.params['A'] @ (style - self.params['smean']).T +
                        np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest

    def train(self, train_data, train_labels,
              test_data=None, test_labels=None,
              channels=None, normalize=True, 
              learning_rate=0.2, n_epochs=10, 
              l2_regularization=1.0, batch_size=8):
        """ train size model with images train_data to estimate linear model from styles to diameters
        
            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            n_epochs: int (default, 10)
                how many times to go through whole training set (taking random patches) for styles for diameter estimation

            l2_regularization: float (default, 1.0)
                regularize linear model from styles to diameters

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)
        """
        batch_size /= 2 # reduce batch_size by factor of 2 to use larger tiles
        batch_size = int(max(1, batch_size))
        self.cp.batch_size = batch_size
        # train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize)
        train_data, train_labels, test_data, test_labels, run_test = reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize)
        if isinstance(self.cp.pretrained_model, list) and len(self.cp.pretrained_model)>1:
            cp_model_path = self.cp.pretrained_model[0]
            self.cp.net.load_model(cp_model_path, cpu=(not self.gpu))
            if not self.torch:
                self.cp.net.collect_params().grad_req = 'null'
        else:
            cp_model_path = self.cp.pretrained_model

        # diam_train = np.array([utils.diameters(lbl)[0] for lbl in train_labels])
        diam_train = np.array([diameters(lbl)[0] for lbl in train_labels])
        if run_test: 
            # diam_test = np.array([utils.diameters(lbl)[0] for lbl in test_labels])
            diam_test = np.array([diameters(lbl)[0] for lbl in test_labels])
        
        nimg = len(train_data)
        styles = np.zeros((n_epochs*nimg, 256), np.float32)
        diams = np.zeros((n_epochs*nimg,), np.float32)
        tic = time.time()
        for iepoch in range(n_epochs):
            iall = np.arange(0,nimg,1,int)
            for ibatch in range(0,nimg,batch_size):
                inds = iall[ibatch:ibatch+batch_size]
                # imgi,lbl,scale = transforms.random_rotate_and_resize(
                imgi,lbl,scale = random_rotate_and_resize(
                            [train_data[i] for i in inds],
                            Y=[train_labels[i].astype(np.int16) for i in inds], scale_range=1, xy=(512,512))
                feat = self.cp.network(imgi)[1]
                styles[inds+nimg*iepoch] = feat
                diams[inds+nimg*iepoch] = np.log(diam_train[inds]) - np.log(self.diam_mean) + np.log(scale)
            del feat
            if (iepoch+1)%2==0:
                print('ran %d epochs in %0.3f sec'%(iepoch+1, time.time()-tic))

        # create model
        smean = styles.mean(axis=0)
        X = ((styles - smean).T).copy()
        ymean = diams.mean()
        y = diams - ymean

        A = np.linalg.solve(X@X.T + l2_regularization*np.eye(X.shape[0]), X @ y)
        ypred = A @ X
        print('train correlation: %0.4f'%np.corrcoef(y, ypred)[0,1])
            
        if run_test:
            nimg_test = len(test_data)
            styles_test = np.zeros((nimg_test, 256), np.float32)
            for i in range(nimg_test):
                styles_test[i] = self.cp._run_net(test_data[i].transpose((1,2,0)))[1]
            diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(self.diam_mean) + ymean)
            diam_test_pred = np.maximum(5., diam_test_pred)
            print('test correlation: %0.4f'%np.corrcoef(diam_test, diam_test_pred)[0,1])

        self.pretrained_size = cp_model_path+'_size.npy'
        self.params = {'A': A, 'smean': smean, 'diam_mean': self.diam_mean, 'ymean': ymean}
        np.save(self.pretrained_size, self.params)
        return self.params


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### plot.py ###################################################################
#################################################################################################################################

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import scipy

# from . import utils, io

def show_segmentation(fig, img, maski, flowi, channels=[0,0], file_name=None):
    """ plot segmentation results (like on website)
    
    Can save each panel of figure with file_name option. Use channels option if
    img input is not an RGB image with 3 channels.
    
    Parameters
    -------------

    fig: matplotlib.pyplot.figure
        figure in which to make plot

    img: 2D or 3D array
        image input into cellpose

    maski: int, 2D array
        for image k, masks[k] output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels

    flowi: int, 2D array 
        for image k, flows[k][0] output from Cellpose.eval (RGB of flows)

    channels: list of int (optional, default [0,0])
        channels used to run Cellpose, no need to use if image is RGB

    file_name: str (optional, default None)
        file name of image, if file_name is not None, figure panels are saved

    """
    ax = fig.add_subplot(1,4,1)
    img0 = img.copy()
    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1,2,0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max()<=50.0:
            img0 = np.uint8(np.clip(img0*255, 0, 1))
    ax.imshow(img0)
    ax.set_title('original image')
    ax.axis('off')

    # outlines = utils.masks_to_outlines(maski)
    outlines = masks_to_outlines(maski)
    overlay = mask_overlay(img0, maski)

    ax = fig.add_subplot(1,4,2)
    outX, outY = np.nonzero(outlines)
    imgout= img0.copy()
    imgout[outX, outY] = np.array([255,75,75])
    ax.imshow(imgout)
    #for o in outpix:
    #    ax.plot(o[:,0], o[:,1], color=[1,0,0], lw=1)
    ax.set_title('predicted outlines')
    ax.axis('off')

    ax = fig.add_subplot(1,4,3)
    ax.imshow(overlay)
    ax.set_title('predicted masks')
    ax.axis('off')

    ax = fig.add_subplot(1,4,4)
    ax.imshow(flowi)
    ax.set_title('predicted cell pose')
    ax.axis('off')

    if file_name is not None:
        save_path = os.path.splitext(file_name)[0]
        # io.imsave(save_path + '_overlay.jpg', overlay)
        imsave(save_path + '_overlay.jpg', overlay)
        # io.imsave(save_path + '_outlines.jpg', imgout)
        imsave(save_path + '_outlines.jpg', imgout)
        # io.imsave(save_path + '_flows.jpg', flowi)
        imsave(save_path + '_flows.jpg', flowi)

def mask_rgb(masks, colors=None):
    """ masks in random rgb colors

    Parameters
    ----------------

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        # colors = utils.rgb_to_hsv(colors)
        colors = rgb_to_hsv(colors)
    
    HSV = np.zeros((masks.shape[0], masks.shape[1], 3), np.float32)
    HSV[:,:,2] = 1.0
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = np.random.rand()*0.5+0.5
        HSV[ipix[0],ipix[1],2] = np.random.rand()*0.5+0.5
    # RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)

    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        # colors = utils.rgb_to_hsv(colors)
        colors = rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    # img = utils.normalize99(img)
    img = normalize99(img)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    # RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def image_to_rgb(img0, channels=[0,0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            # img[:,:,i] = utils.normalize99(img[:,:,i])
            img[:,:,i] = normalize99(img[:,:,i])
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

def interesting_patch(mask, bsize=130):
    """ get patch of size bsize x bsize with most masks """
    Ly,Lx = mask.shape
    m = np.float32(mask>0)
    m = gaussian_filter(m, bsize/2)
    y,x = np.unravel_index(np.argmax(m), m.shape)
    ycent = max(bsize//2, min(y, Ly-bsize//2))
    xcent = max(bsize//2, min(x, Lx-bsize//2))
    patch = [np.arange(ycent-bsize//2, ycent+bsize//2, 1, int),
             np.arange(xcent-bsize//2, xcent+bsize//2, 1, int)]
    return patch

def disk(med, r, Ly, Lx):
    """ returns pixels of disk with radius r and center med """
    yy, xx = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int),
                         indexing='ij')
    inds = ((yy-med[0])**2 + (xx-med[1])**2)**0.5 <= r
    y = yy[inds].flatten()
    x = xx[inds].flatten()
    return y,x


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
################################################## resnet_style.py ##############################################################
#################################################################################################################################
'''
from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np

nfeat = 128
rssz  = [3, 3, 3, 3, 3]
rssz2 = [3, 3, 3, 3, 3]
rsszf = [1]


def total_variation_loss(x):
    """ regularize convolutional masks (not currently in use) """
    a = nd.square(x[:, :, :-1, :-1] - x[:, :, 1:, :-1])
    b = nd.square(x[:, :, :-1, :-1] - x[:, :, :-1, 1:])
    return nd.sum(nd.mean(nd.power(a + b, 1.25), axis=(2,3)))

def convbatchrelu(nconv, rssz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.Conv2D(nconv, kernel_size=rssz, padding=rssz//2),
                nn.BatchNorm(axis=1),
                nn.Activation('relu'),
        )
    return conv

def batchconv(nconv, rssz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.BatchNorm(axis=1),
                nn.Activation('relu'),
                nn.Conv2D(nconv, kernel_size=rssz, padding=rssz//2),
        )
    return conv

def batchconv0(nconv, rssz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.BatchNorm(axis=1),
                nn.Conv2D(nconv, kernel_size=rssz, padding=rssz//2),
        )
    return conv

class resdown(nn.HybridBlock):
    def __init__(self, nconv, **kwargs):
        super(resdown, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            for t in range(4):
                self.conv.add( batchconv(nconv, 3))
            self.proj  = batchconv0(nconv, 1)

    def hybrid_forward(self, F, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.HybridBlock):
    def __init__(self, nconv, **kwargs):
        super(convdown, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            for t in range(2):
                self.conv.add(batchconv(nconv, 3))

    def hybrid_forward(self, F, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.HybridBlock):
    def __init__(self, nbase, residual_on=True, **kwargs):
        super(downsample, self).__init__(**kwargs)
        with self.name_scope():
            self.down = nn.HybridSequential()
            for n in range(len(nbase)):
                if residual_on:
                    self.down.add(resdown(nbase[n]))
                else:
                    self.down.add(convdown(nbase[n]))

    def hybrid_forward(self, F, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = F.Pooling(xd[n-1], kernel=(2,2), stride=(2,2), pool_type='max')
            else:
                y = x
            xd.append(self.down[n](y))
        return xd

class batchconvstyle(nn.HybridBlock):
    def __init__(self, nconv, concatenation=False, **kwargs):
        super(batchconvstyle, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = batchconv(nconv, 3)
            if concatenation:
                self.full = nn.Dense(nconv*2)
            else:
                self.full = nn.Dense(nconv)
            self.concatenation = concatenation

    def hybrid_forward(self, F, style, x, y=None):
        if y is not None:
            if self.concatenation:
                x = F.concat(y, x, dim=1)
            else:
                x = x + y
        feat = self.full(style)
        y = F.broadcast_add(x, feat.expand_dims(-1).expand_dims(-1))
        y = self.conv(y)
        return y

class convup(nn.HybridBlock):
    def __init__(self, nconv, concatenation=False, **kwargs):
        super(convup, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(batchconv(nconv, 3))
            self.conv.add(batchconvstyle(nconv, concatenation))
            
    def hybrid_forward(self, F, x, y, style):
        x = self.conv[0](x)
        x = self.conv[1](style, x, y)
        return x


class resup(nn.HybridBlock):
    def __init__(self, nconv, concatenation=False, **kwargs):
        super(resup, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(batchconv(nconv,3))
            self.conv.add(batchconvstyle(nconv, concatenation))
            self.conv.add(batchconvstyle(nconv))
            self.conv.add(batchconvstyle(nconv))
            self.proj  = batchconv0(nconv, 1)

    def hybrid_forward(self, F, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x

class upsample(nn.HybridBlock):
    def __init__(self, nbase, residual_on=True, concatenation=False, **kwargs):
        super(upsample, self).__init__(**kwargs)
        with self.name_scope():
            self.up = nn.HybridSequential()
            for n in range(len(nbase)):
                if residual_on:
                    self.up.add(resup(nbase[n], concatenation=concatenation))
                else:
                    self.up.add(convup(nbase[n], concatenation=concatenation))

    def hybrid_forward(self, F, style, xd):
        x= self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up)-2,-1,-1):
            x= F.UpSampling(x, scale=2, sample_type='nearest')
            x = self.up[n](x, xd[n], style)
        return x

class make_style(nn.HybridBlock):
    def __init__(self,  **kwargs):
        super(make_style, self).__init__(**kwargs)
        with self.name_scope():
            self.pool_all = nn.GlobalAvgPool2D()
            self.flatten = nn.Flatten()

    def hybrid_forward(self, F, x0):
        style = self.pool_all(x0)
        style = self.flatten(style)
        style = F.broadcast_div(style , F.sum(style**2, axis=1).expand_dims(1)**.5)

        return style

class rsCPnet(gluon.HybridBlock):
    def __init__(self, nbase, nout, residual_on=True, style_on=True, concatenation=False, **kwargs):
        super(rsCPnet, self).__init__(**kwargs)
        with self.name_scope():
            self.nbase = nbase
            self.downsample = downsample(nbase, residual_on=residual_on)
            self.upsample = upsample(nbase, residual_on=residual_on, concatenation=concatenation)
            self.output = batchconv(nout, 1)
            self.make_style = make_style()
            self.style_on = style_on

    def hybrid_forward(self, F, data):
        #data     = self.conv1(data)
        T0    = self.downsample(data)
        style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0 
        T0    = self.upsample(style, T0)
        T1    = self.output(T0)

        return T1, style0, T0

    def save_model(self, filename):
        self.save_parameters(filename)

    def load_model(self, filename, cpu=None):
        self.load_parameters(filename)


'''
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
################################################## resnet_torch.py ##############################################################
#################################################################################################################################

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime


# from . import transforms, io, dynamics, utils

rtsz = 3

def convbatchrelu(in_channels, out_channels, rtsz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, rtsz, padding=rtsz//2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )  

def batchconv(in_channels, out_channels, rtsz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, rtsz, padding=rtsz//2),
    )  

def batchconv0(in_channels, out_channels, rtsz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, rtsz, padding=rtsz//2),
    )  

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, rtsz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj  = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, rtsz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, rtsz))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, rtsz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, rtsz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, rtsz))
                
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class downsample(nn.Module):
    def __init__(self, nbase, rtsz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase)-1):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], rtsz))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], rtsz))
            
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
    
class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, rtsz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        self.conv = batchconv(in_channels, out_channels, rtsz)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels*2)
        else:
            self.full = nn.Linear(style_channels, out_channels)
        
    def forward(self, style, x, mkldnn=False):
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, rtsz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, rtsz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, rtsz, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, rtsz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, rtsz))
        self.proj  = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x
    
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, rtsz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, rtsz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, rtsz, concatenation=concatenation))
        
    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x) + y)
        return x
    
class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        #style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2],x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, rtsz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d'%(n-1), 
                    resup(nbase[n], nbase[n-1], nbase[-1], rtsz, concatenation))
            else:
                self.up.add_module('conv_up_%d'%(n-1), 
                    convup(nbase[n], nbase[n-1], nbase[-1], rtsz, concatenation))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up)-2,-1,-1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x
    
class rtCPnet(nn.Module):
    def __init__(self, nbase, nout, rtsz, residual_on=True, 
                 style_on=True, concatenation=False, mkldnn=False):
        super(rtCPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.rtsz = rtsz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, rtsz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, rtsz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.style_on = style_on
        
    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0    = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense()) 
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        T1    = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()    
            T1 = T1.to_dense()    
        return T1, style0, T0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            self.load_state_dict(torch.load(filename))
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.rtsz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn)
            self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### transforms.py #############################################################
#################################################################################################################################

#from suite2p import nonrigid
import numpy as np
import warnings
import cv2

def _taper_mask(ly=224, lx=224, sig=7.5):
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize//2-ly//2 : bsize//2+ly//2+ly%2, 
                bsize//2-lx//2 : bsize//2+lx//2+lx%2]
    return mask

def unaugment_tiles(y, unet=False):
    """ reverse test-time augmentations for averaging

    Parameters
    ----------

    y: float32
        array that's ntiles_y x ntiles_x x chan x Ly x Lx where chan = (dY, dX, cell prob)

    unet: bool (optional, False)
        whether or not unet output or cellpose output
    
    Returns
    -------

    y: float32

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j%2==0 and i%2==1:
                y[j,i] = y[j,i, :,::-1, :]
                if not unet:
                    y[j,i,0] *= -1
            elif j%2==1 and i%2==0:
                y[j,i] = y[j,i, :,:, ::-1]
                if not unet:
                    y[j,i,1] *= -1
            elif j%2==1 and i%2==1:
                y[j,i] = y[j,i, :,::-1, ::-1]
                if not unet:
                    y[j,i,0] *= -1
                    y[j,i,1] *= -1
    return y

def average_tiles(y, ysub, xsub, Ly, Lx):
    """ average results of network over tiles

    Parameters
    -------------

    y: float, [ntiles x nclasses x bsize x bsize]
        output of cellpose network for each tile

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of pre-tiled image in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of pre-tiled image in X (may be larger than original image if
        image size is less than bsize)

    Returns
    -------------

    yf: float32, [nclasses x Ly x Lx]
        network output averaged over tiles

    """
    Navg = np.zeros((Ly,Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf

def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """ make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    
    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize-Ly, Lx))), axis=1)
            Ly = bsize
        if Lx<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize-Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly-bsize, ny).astype(int)
        xstart = np.linspace(0, Lx-bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsize])
                xsub.append([xstart[i], xstart[i]+bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j%2==0 and i%2==1:
                    IMG[j,i] = IMG[j,i, :,::-1, :]
                elif j%2==1 and i%2==0:
                    IMG[j,i] = IMG[j,i, :,:, ::-1]
                elif j%2==1 and i%2==1:
                    IMG[j,i] = IMG[j,i,:, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly<=bsize else int(np.ceil((1.+2*tile_overlap) * Ly / bsize))
        nx = 1 if Lx<=bsize else int(np.ceil((1.+2*tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly-bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx-bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsizeY])
                xsub.append([xstart[i], xstart[i]+bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
        
    return IMG, ysub, xsub, Ly, Lx

def normalize99(img):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def reshape(data, channels=[0,0], chan_first=False):
    """ reshape data using channels

    Parameters
    ----------

    data : numpy array that's (Z x ) Ly x Lx x nchan
        if data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    invert : bool
        invert intensities

    Returns
    -------
    data : numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False)

    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:,:,np.newaxis]
    elif data.shape[0]<8 and data.ndim==3:
        data = np.transpose(data, (1,2,0))    

    # use grayscale image
    if data.shape[-1]==1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0]==0:
            data = data.mean(axis=-1)
            data = np.expand_dims(data, axis=-1)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0]-1]
            if channels[1] > 0:
                chanid.append(channels[1]-1)
            data = data[...,chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[...,i]) == 0.0:
                    if i==0:
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn("'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1]==1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim==4:
            data = np.transpose(data, (3,0,1,2))
        else:
            data = np.transpose(data, (2,0,1))
    return data

def normalize_img(img, axis=-1, invert=False):
    """ normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim<3:
        raise ValueError('Image needs to have at least 3 dimensions')

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if np.ptp(img[k]) > 0.0:
            img[k] = normalize99(img[k])
            if invert:
                img[k] = -1*img[k] + 1   
    img = np.moveaxis(img, 0, axis)
    return img

def reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize):
    """ check sizes and reshape train and test data for training """
    nimg = len(train_data)
    # check that arrays are correct size
    if nimg != len(train_labels):
        raise ValueError('train data and labels not same length')
        return
    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        raise ValueError('training data or labels are not at least two-dimensional')
        return

    if train_data[0].ndim > 3:
        raise ValueError('training data is more than three-dimensional (should be 2D or 3D array)')
        return

    # check if test_data correct length
    if not (test_data is not None and test_labels is not None and
            len(test_data) > 0 and len(test_data)==len(test_labels)):
        test_data = None

    # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
    train_data, test_data, run_test = reshape_and_normalize_data(train_data, test_data=test_data, 
                                                                 channels=channels, normalize=normalize)

    if train_data is None:
        raise ValueError('training data do not all have the same number of channels')
        return

    if not run_test:
        print('NOTE: test data not provided OR labels incorrect OR not same number of channels as train data')
        test_data, test_labels = None, None

    return train_data, train_labels, test_data, test_labels, run_test

def reshape_and_normalize_data(train_data, test_data=None, channels=None, normalize=True):
    """ inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel

    Parameters
    --------------

    train_data: list of ND-arrays, float
        list of training images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    channels: list of int of length 2 (optional, default None)
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    normalize: bool (optional, True)
        normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

    Returns
    -------------

    train_data: list of ND-arrays, float
        list of training images of size [2 x Ly x Lx]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [2 x Ly x Lx]

    run_test: bool
        whether or not test_data was correct size and is useable during training

    """

    # if training data is less than 2D
    nimg = len(train_data)
    if channels is not None:
        train_data = [reshape(train_data[n], channels=channels, chan_first=True) for n in range(nimg)]
    if train_data[0].ndim < 3:
        train_data = [train_data[n][:,:,np.newaxis] for n in range(nimg)]
    elif train_data[0].shape[-1] < 8:
        print('NOTE: assuming train_data provided as Ly x Lx x nchannels, transposing axes to put channels first')
        train_data = [np.transpose(train_data[n], (2,0,1)) for n in range(nimg)]
    nchan = [train_data[n].shape[0] for n in range(nimg)]
    if nchan.count(nchan[0]) != len(nchan):
        return None, None, None
    nchan = nchan[0]

    # check for valid test data
    run_test = False
    if test_data is not None:
        nimgt = len(test_data)
        if channels is not None:
            test_data = [reshape(test_data[n], channels=channels, chan_first=True) for n in range(nimgt)]
        if test_data[0].ndim==2:
            if nchan==1:
                run_test = True
                test_data = [test_data[n][np.newaxis,:,:] for n in range(nimgt)]
        elif test_data[0].ndim==3:
            if test_data[0].shape[-1] < 8:
                print('NOTE: assuming test_data provided as Ly x Lx x nchannels, transposing axes to put channels first')
                test_data = [np.transpose(test_data[n], (2,0,1)) for n in range(nimgt)]
            nchan_test = [test_data[n].shape[0] for n in range(nimgt)]
            if nchan_test.count(nchan_test[0]) != len(nchan_test):
                run_test = False
            elif test_data[0].shape[0]==nchan:
                run_test = True
    
    if normalize:
        train_data = [normalize_img(train_data[n], axis=0) for n in range(nimg)]
        if run_test:
            test_data = [normalize_img(test_data[n], axis=0) for n in range(nimgt)]

    return train_data, test_data, run_test

def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [y x x x nchan] or [Lz x y x x x nchan]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

    Returns
    --------------

    imgs: ND-array 
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if Ly is None and rsz is None:
        raise ValueError('must give size to resize to or factor to use for resizing')

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Ly = int(img0.shape[-3] * rsz[-2])
        Lx = int(img0.shape[-2] * rsz[-1])
    
    Lx = int(round(Lx))
    Ly = int(round(Ly))

    if img0.ndim==4:
        imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs

def pad_image_ND(img0, div=16, extra = 1):
    """ pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)

    Parameters
    -------------

    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]

    div: int (optional, default 16)

    Returns
    --------------

    I: ND-array
        padded image

    ysub: array, int
        yrange of pixels in I corresponding to img0

    xsub: array, int
        xrange of pixels in I corresponding to img0

    """
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1+Ly)
    xsub = np.arange(ypad1, ypad1+Lx)
    return I, ysub, xsub

def random_rotate_and_resize(X, Y=None, scale_range=1., xy = (224,224), 
                             do_flip=True, rescale=None, unet=False):
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow]. 
            If unet, second channel is dist_to_bound.

        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()

        xy: tuple, int (optional, default (224,224))
            size of transformed images to return

        do_flip: bool (optional, default True)
            whether or not to flip images horizontally

        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations

        unet: bool (optional, default False)

        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]

        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]

        scale: array, float
            amount each image was resized by

    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim>2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y is not None:
        if Y[0].ndim>2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)

    scale = np.zeros(nimg, np.float32)
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        # generate random augmentation parameters
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        scale[n] = (1-scale_range/2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[n] *= 1. / rescale[n]
        dxy = np.maximum(0, np.array([Lx*scale[n]-xy[1],Ly*scale[n]-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy

        # create affine transform
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale[n]*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale[n]*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:]

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[2] = -labels[2]

        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n,k] = I

        if Y is not None:
            for k in range(nt):
                if k==0:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_NEAREST)
                else:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)

            if nt > 1 and not unet:
                v1 = lbl[n,2].copy()
                v2 = lbl[n,1].copy()
                lbl[n,1] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
                lbl[n,2] = (v1 * np.cos(-theta) + v2*np.sin(-theta))

    return imgi, lbl, scale


def _X2zoom(img, X2=1):
    """ zoom in image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    Returns
    -------
    img : numpy array that's Ly x Lx

    """
    ny,nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2**X2)), int(ny * (2**X2))))
    return img

def _image_resizer(img, resize=512, to_uint8=False):
    """ resize image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    resize : int
        max size of image returned

    to_uint8 : bool
        convert image to uint8

    Returns
    -------
    img : numpy array that's Ly x Lx, Ly,Lx<resize

    """
    ny,nx = img.shape[:2]
    if to_uint8:
        if img.max()<=255 and img.min()>=0 and img.max()>1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### utils.py ##################################################################
#################################################################################################################################

import os, warnings, time, tempfile, datetime, pathlib, shutil, subprocess
from tqdm import tqdm
from urllib.request import urlopen
from urllib.parse import urlparse
import cv2
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
from scipy.spatial import ConvexHull
import numpy as np
import colorsys

# from . import metrics

def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv

def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb


def distance_to_boundary(masks):
    """ get distance to boundary of mask pixels
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    dist_to_bound: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('distance_to_boundary takes 2D or 3D array, not %dD array'%masks.ndim)
    dist_to_bound = np.zeros(masks.shape, np.float64)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            dist_to_bound[i] = distance_to_boundary(masks[i])
        return dist_to_bound
    else:
        slices = find_objects(masks)
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T  
                ypix, xpix = np.nonzero(mask)
                min_dist = ((ypix[:,np.newaxis] - pvr)**2 + 
                            (xpix[:,np.newaxis] - pvc)**2).min(axis=1)
                dist_to_bound[ypix + sr.start, xpix + sc.start] = min_dist
        return dist_to_bound

def masks_to_edges(masks, threshold=1.0):
    """ get edges of masks as a 0-1 array 
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    edges: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are edge pixels

    """
    dist_to_bound = distance_to_boundary(masks)
    edges = (dist_to_bound < threshold) * (masks > 0)
    return edges

def masks_to_outlines(masks):
    """ get outlines of masks as a 0-1 array 
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    outlines = np.zeros(masks.shape, np.bool)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                vr, vc = pvr + sr.start, pvc + sc.start 
                outlines[vr, vc] = 1
        return outlines

def outlines_list(masks):
    """ get outlines of masks as a list to loop over for plotting """
    outpix=[]
    for n in np.unique(masks)[1:]:
        mn = masks==n
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix)>4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0,2)))
    return outpix

def get_perimeter(points):
    """ perimeter of points - npoints x ndim """
    if points.shape[0]>4:
        points = np.append(points, points[:1], axis=0)
        return ((np.diff(points, axis=0)**2).sum(axis=1)**0.5).sum()
    else:
        return 0

def get_mask_compactness(masks):
    perimeters = get_mask_perimeters(masks)
    #outlines = masks_to_outlines(masks)
    #perimeters = np.unique(outlines*masks, return_counts=True)[1][1:]
    npoints = np.unique(masks, return_counts=True)[1][1:]
    areas = npoints
    compactness =  4 * np.pi * areas / perimeters**2
    compactness[perimeters==0] = 0
    compactness[compactness>1.0] = 1.0
    return compactness

def get_mask_perimeters(masks):
    """ get perimeters of masks """
    perimeters = np.zeros(masks.max())
    for n in range(masks.max()):
        mn = masks==(n+1)
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)[-2]
            #cmax = np.argmax([c.shape[0] for c in contours])
            #perimeters[n] = get_perimeter(contours[cmax].astype(int).squeeze())
            perimeters[n] = np.array([get_perimeter(c.astype(int).squeeze()) for c in contours]).sum()

    return perimeters

def circleMask(d0):
    """ creates array with indices which are the radius of that x,y point
        inputs:
            d0 (patch of (-d0,d0+1) over which radius computed
        outputs:
            rs: array (2*d0+1,2*d0+1) of radii
            dx,dy: indices of patch
    """
    dx  = np.tile(np.arange(-d0[1],d0[1]+1), (2*d0[0]+1,1))
    dy  = np.tile(np.arange(-d0[0],d0[0]+1), (2*d0[1]+1,1))
    dy  = dy.transpose()

    rs  = (dy**2 + dx**2) ** 0.5
    return rs, dx, dy

def get_mask_stats(masks_true):
    mask_perimeters = get_mask_perimeters(masks_true)

    # disk for compactness
    rs,dy,dx = circleMask(np.array([100, 100]))
    rsort = np.sort(rs.flatten())

    # area for solidity
    npoints = np.unique(masks_true, return_counts=True)[1][1:]
    areas = npoints - mask_perimeters / 2 - 1
    
    compactness = np.zeros(masks_true.max())
    convexity = np.zeros(masks_true.max())
    solidity = np.zeros(masks_true.max())
    convex_perimeters = np.zeros(masks_true.max())
    convex_areas = np.zeros(masks_true.max())
    for ic in range(masks_true.max()):
        points = np.array(np.nonzero(masks_true==(ic+1))).T
        if len(points)>15 and mask_perimeters[ic] > 0:
            med = np.median(points, axis=0)
            # compute compactness of ROI
            r2 = ((points - med)**2).sum(axis=1)**0.5
            compactness[ic] = (rsort[:r2.size].mean() + 1e-10) / r2.mean()
            try:
                hull = ConvexHull(points)
                convex_perimeters[ic] = hull.area
                convex_areas[ic] = hull.volume
            except:
                convex_perimeters[ic] = 0
                
    convexity[mask_perimeters>0.0] = (convex_perimeters[mask_perimeters>0.0] / 
                                      mask_perimeters[mask_perimeters>0.0])
    solidity[convex_areas>0.0] = (areas[convex_areas>0.0] / 
                                     convex_areas[convex_areas>0.0])
    convexity = np.clip(convexity, 0.0, 1.0)
    solidity = np.clip(solidity, 0.0, 1.0)
    compactness = np.clip(compactness, 0.0, 1.0)
    return convexity, solidity, compactness

def get_masks_unet(output, cell_threshold=0, boundary_threshold=0):
    """ create masks using cell probability and cell boundary """
    cells = (output[...,1] - output[...,0])>cell_threshold
    selem = generate_binary_structure(cells.ndim, connectivity=1)
    labels, nlabels = label(cells, selem)

    if output.shape[-1]>2:
        slices = find_objects(labels)
        dists = 10000*np.ones(labels.shape, np.float32)
        mins = np.zeros(labels.shape, np.int32)
        borders = np.logical_and(~(labels>0), output[...,2]>boundary_threshold)
        pad = 10
        for i,slc in enumerate(slices):
            if slc is not None:
                slc_pad = tuple([slice(max(0,sli.start-pad), min(labels.shape[j], sli.stop+pad))
                                    for j,sli in enumerate(slc)])
                msk = (labels[slc_pad] == (i+1)).astype(np.float32)
                msk = 1 - gaussian_filter(msk, 5)
                dists[slc_pad] = np.minimum(dists[slc_pad], msk)
                mins[slc_pad][dists[slc_pad]==msk] = (i+1)
        labels[labels==0] = borders[labels==0] * mins[labels==0]
        
    masks = labels
    shape0 = masks.shape
    _,masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
    return masks

def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        # iou = metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
        iou = _intersection_over_union(masks[i+1], masks[i])[1:,1:]
        iou[iou < stitch_threshold] = 0.0
        iou[iou < iou.max(axis=0)] = 0.0
        istitch = iou.argmax(axis=1) + 1
        ino = np.nonzero(iou.max(axis=1)==0.0)[0]
        istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
        mmax += len(ino)
        istitch = np.append(np.array(0), istitch)
        masks[i+1] = istitch[masks[i+1]]
    return masks

def diameters(masks):
    """ get median 'diameter' of masks """
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return md, counts**0.5

def radius_distribution(masks, bins):
    unique, counts = np.unique(masks, return_counts=True)
    counts = counts[unique!=0]
    nb, _ = np.histogram((counts**0.5)*0.5, bins)
    nb = nb.astype(np.float32)
    if nb.sum() > 0:
        nb = nb / nb.sum()
    md = np.median(counts**0.5)*0.5
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return nb, md, (counts**0.5)/2

def size_distribution(masks):
    counts = np.unique(masks, return_counts=True)[1][1:]
    return np.percentile(counts, 25) / np.percentile(counts, 75)

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def process_cells(M0, npix=20):
    unq, ic = np.unique(M0, return_counts=True)
    for j in range(len(unq)):
        if ic[j]<npix:
            M0[M0==unq[j]] = 0
    return M0


def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    
    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('fill_holes_and_remove_small_masks takes 2D or 3D array, not %dD array'%masks.ndim)
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:    
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


#################################################################################################################################
##################################################### ____.py ###################################################################
#################################################################################################################################
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


def PyInit_Compiled():
    pass
