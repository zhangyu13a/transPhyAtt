
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from YoloV3.misc_functions import *
from torchvision.ops import nms
from torchvision.utils import save_image
import torchvision.transforms as transforms

import warnings
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from collections.abc import Sequence
from typing import Tuple, List, Optional
import numbers
from torch import Tensor

class Attention(object):

    def __init__(self, net, ori_shape, final_shape, yolo_decodes,num_classes, conf_thres, nms_thres):
        self.net = net
        self.ori_shape = ori_shape
        self.final_shape = final_shape
        self.feature = list()
        self.gradient = list()
        self.net.eval()
        self.yolo_decodes=yolo_decodes
        self.num_classes=num_classes
        self.conf_thres=conf_thres
        self.nms_thres=nms_thres
        self.handlers = []

    def _get_features_hook(self, module, input, output):
        self.feature.append(output)
        # print("feature shape:{}".format(output.size()))
   
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient.append(output_grad[0] )
        # print('output_grad[0].shape=',output_grad[0].shape)

    def _register_hook(self):
        self.feature = list()#clear self.feature and self.gradient 
        self.gradient = list()
        self.handlers = [] #clear self.handlers

        for i, module in enumerate(self.net.module.backbone.layer5.residual_1._modules.items()):
            if module[1] == self.net.module.backbone.layer5.residual_1.conv2:
                self.handlers.append(module[1].register_forward_hook(self._get_features_hook))
                self.handlers.append(module[1].register_backward_hook(self._get_grads_hook))

        for i, module in enumerate(self.net.module.backbone.layer4.residual_5._modules.items()):
            if module[1] == self.net.module.backbone.layer4.residual_5.conv2:
                self.handlers.append(module[1].register_forward_hook(self._get_features_hook))
                self.handlers.append(module[1].register_backward_hook(self._get_grads_hook))

        for i, module in enumerate(self.net.module.backbone.layer3.residual_2._modules.items()):
            if module[1] == self.net.module.backbone.layer3.residual_2.conv2:
                self.handlers.append(module[1].register_forward_hook(self._get_features_hook))
                self.handlers.append(module[1].register_backward_hook(self._get_grads_hook))
        
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
            torch.cuda.empty_cache()

    def __call__(self, inputs, index=0,retain_graph=True):
        
        img_ori=inputs['image']
        Hflip=transforms.RandomHorizontalFlip(p=1)

        '''1st compound transformation ：trans->scale'''
        # translation
        randomAffine1=myRandomAffine(degrees=(0,0),translate=(0.1,0.1))
        img_randomAffine1,translations_b1,_=randomAffine1(img_ori.clone())            
        # inverse translation
        inv_randomAffine_b1=myRandomAffine(degrees=(0,0),inv_translate=(-translations_b1[0],-translations_b1[1]))
        #scaling
        randomAffine_scale1=myRandomAffine(degrees=(0,0),scale=(0.8,1.2),interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale1,_,scale_factor_b1=randomAffine_scale1(img_randomAffine1)      
        # inverse scaling
        inv_randomScale_b1=myRandomAffine(degrees=(0,0),scale=(1/scale_factor_b1,1/scale_factor_b1),interpolation=InterpolationMode.BILINEAR)
        
        '''2nd：Hflip->trans->scale'''
        img_flip2=Hflip(img_ori.clone())
        # translation
        randomAffine2=myRandomAffine(degrees=(0,0),translate=(0.1,0.1))
        img_randomAffine2,translations_b2,_=randomAffine2(img_flip2)              
        # inverse translation
        inv_randomAffine_b2=myRandomAffine(degrees=(0,0),inv_translate=(-translations_b2[0],-translations_b2[1]))
        #scaling
        randomAffine_scale2=myRandomAffine(degrees=(0,0),scale=(0.8,1.2),interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale2,_,scale_factor_b2=randomAffine_scale2(img_randomAffine2)       
        # inverse scaling
        inv_randomScale_b2=myRandomAffine(degrees=(0,0),scale=(1/scale_factor_b2,1/scale_factor_b2),interpolation=InterpolationMode.BILINEAR)
        
        '''3rd：trans->scale'''  
        # translation
        randomAffine3=myRandomAffine(degrees=(0,0),translate=(0.1,0.1))
        img_randomAffine3,translations_b3,_=randomAffine3(img_ori.clone())            
        # inverse translation
        inv_randomAffine_b3=myRandomAffine(degrees=(0,0),inv_translate=(-translations_b3[0],-translations_b3[1]))
        #scaling
        randomAffine_scale3=myRandomAffine(degrees=(0,0),scale=(0.8,1.2),interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale3,_,scale_factor_b3=randomAffine_scale3(img_randomAffine3)    
        #inverse scaling
        inv_randomScale_b3=myRandomAffine(degrees=(0,0),scale=(1/scale_factor_b3,1/scale_factor_b3),interpolation=InterpolationMode.BILINEAR)
           
        '''4th：Hflip->trans->scale
        img_flip4=Hflip(img_ori.clone())
        randomAffine4=myRandomAffine(degrees=(0,0),translate=(0.1,0.1))
        img_randomAffine4,translations_b4,_=randomAffine4(img_flip4)            
        inv_randomAffine_b4=myRandomAffine(degrees=(0,0),inv_translate=(-translations_b4[0],-translations_b4[1]))
        randomAffine_scale4=myRandomAffine(degrees=(0,0),scale=(0.8,1.2),interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale4,_,scale_factor_b4=randomAffine_scale4(img_randomAffine4)     
        inv_randomScale_b4=myRandomAffine(degrees=(0,0),scale=(1/scale_factor_b4,1/scale_factor_b4),interpolation=InterpolationMode.BILINEAR)
        '''

        img_ensemble=torch.cat((img_randomAffineScale1,img_randomAffineScale2,img_randomAffineScale3),dim=0)      
        outputs = self.net(img_ensemble)
        output_list = []
        for k in range(3):
            output_list.append(self.yolo_decodes[k](outputs[k]))
        output = torch.cat(output_list, 1) #  BS×nx6   (xywh，obj_conf, cls_conf)
        bb=output[:,:,4]+output[:,:,5] 
        scores=torch.max(bb,dim=1).values 
        one_hot_output = torch.FloatTensor(scores.size()[-1]).zero_().cuda()         
        one_hot_output[:] = 1
        self.net.zero_grad()
        scores.backward(gradient=one_hot_output, retain_graph = retain_graph)

        '''-------------------multi-scale attention calculating----------------------'''
        BS=img_ori.shape[0]
        grad_cam_resize_list=list() 
        cam_ori_list=list() 
        cam_ori_norm_list=list() 
        for kk in range(len(self.gradient)):#the larger kk, the lower resolution
            assert len(self.feature)==len(self.gradient) or len(self.feature)==2*len(self.gradient) \
                or len(self.feature)==3*len(self.gradient)or len(self.feature)==4*len(self.gradient),'Error! '
            cam_weigt=torch.mean(self.gradient[len(self.gradient)-kk-1].clone().detach(),dim=[2,3],keepdim=True)-torch.tensor([1]).cuda()
            aa=torch.sum((cam_weigt*self.feature[kk]),dim=1)#
            bb=torch.relu(aa).unsqueeze(1)
            bb_min=torch.min(torch.min(bb.clone().detach(),dim=3).values,dim=2).values
            bb_min = bb_min.unsqueeze(-1)
            bb_min = bb_min.expand(-1, -1, bb.size(2))
            bb_min = bb_min.unsqueeze(-1)
            bb_min = bb_min.expand(-1, -1,-1, bb.size(3))
            bb_max=torch.max(torch.max(bb.clone().detach(),dim=3).values,dim=2).values
            bb_max = bb_max.unsqueeze(-1)
            bb_max = bb_max.expand(-1, -1, bb.size(2))
            bb_max = bb_max.unsqueeze(-1)
            bb_max = bb_max.expand(-1, -1,-1, bb.size(3))
            grad_cam=(bb-bb_min)/(bb_max-bb_min)
            # save_image(grad_cam[0,:,:,:].unsqueeze(0).cpu().detach(), 'TotalImg_120.png')
            cam_ori=aa 
            cam_ori_norm=grad_cam
            transform = transforms.Compose([transforms.Resize(size=(self.ori_shape[1],self.ori_shape[0]))])
            grad_cam_resize=transform(grad_cam)
            
            num=int(img_ensemble.shape[0]/BS)#
            
            #----------------inverse transformation on the attention-------------------#
            for ss in range(num): 
                mask_tmp=grad_cam_resize[BS*ss:BS*(ss+1)]
                if ss==0:#   
                    # mask0=mask_tmp
                    mask_tmp1,_,_=inv_randomScale_b1(mask_tmp) 
                    mask_tmp2,_,_=inv_randomAffine_b1(mask_tmp1)               
                    mask0=mask_tmp2
                elif ss==1:
                    mask_tmp1,_,_=inv_randomScale_b2(mask_tmp)
                    mask_tmp2,_,_=inv_randomAffine_b2(mask_tmp1)  
                    mask1=Hflip(mask_tmp2)
                    
                elif ss==2:
                    mask_tmp1,_,_=inv_randomScale_b3(mask_tmp)
                    mask_tmp2,_,_=inv_randomAffine_b3(mask_tmp1)    
                    mask2=mask_tmp2

                # elif ss==3:
                #     mask_tmp1,_,_=inv_randomScale_b4(mask_tmp)
                #     mask_tmp2,_,_=inv_randomAffine_b4(mask_tmp1) 
                #     mask3=Hflip(mask_tmp2)                         
                else:
                    raise ValueError("Error.")            
 
            for ll in range(BS):
                final_scores=1./num
                grad_cam_resize_ensem_tmp=mask0[ll].unsqueeze(0)*final_scores+mask1[ll].unsqueeze(0)*final_scores+\
                                            mask2[ll].unsqueeze(0)*final_scores#+mask3[ll].unsqueeze(0)*final_scores              
                if ll==0:
                    grad_cam_resize_ensem=grad_cam_resize_ensem_tmp
                else:
                    grad_cam_resize_ensem=torch.cat((grad_cam_resize_ensem,grad_cam_resize_ensem_tmp),dim=0)              
            grad_cam_resize_list.append(grad_cam_resize_ensem)
            cam_ori_list.append(cam_ori[:BS]) 
            cam_ori_norm_list.append(cam_ori_norm[:BS])
            
        mm=grad_cam_resize_list[1][-1,0,:,:].cpu().detach()
        nn=np.uint8(mm * 255)
        heatmap = cv2.applyColorMap(nn, cv2.COLORMAP_JET)

        test_img = inputs['image'].clone()[-1,:,:,:].cpu().detach()
        # [0,1]->[0,255]，CHW->HWC，->cv2
        test_img=test_img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        superimposed_img = heatmap * 0.6 + test_img*0.4
        cv2.imwrite('./attention_map_sample.jpg', superimposed_img)#layer5
        return grad_cam_resize_list,cam_ori_list,cam_ori_norm_list
        
class myRandomAffine(torch.nn.Module):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
            If input is PIL Image, the options is only available for ``Pillow>=5.0.0``.
        fillcolor (sequence or number, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``fill`` parameter instead.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(#adding translate and inv_translate
        self, degrees, translate=None,inv_translate=None,scale=None,inv_scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0,
        fillcolor=None, resample=None
    ):
        super().__init__()

        if fillcolor is not None:
            warnings.warn(
                "Argument fillcolor is deprecated and will be removed since v0.10.0. Please, use fill instead"
            )
            fill = fillcolor

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2, ))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        #new
        if inv_translate is not None:
            _check_sequence_input(inv_translate, "inv_translate", req_sizes=(2, ))
        self.inv_translate = inv_translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2, ))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        #new
        if inv_scale is not None:
            _check_sequence_input(inv_scale, "inv_scale", req_sizes=(2, ))
        self.inv_scale = inv_scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            inv_translate: Optional[List[float]],#
            scale_ranges: Optional[List[float]],
            inv_scale_ranges: Optional[List[float]],#
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        elif inv_translate is not None:#
            translations=inv_translate
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        elif inv_scale_ranges is not None:#
            scale=inv_scale_ranges
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate,self.inv_translate, self.scale,self.inv_scale, self.shear, img_size)
        angle, translations, scale, shear=ret #
        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill),translations, scale#

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.interpolation != InterpolationMode.NEAREST:
            s += ', interpolation={interpolation}'
        if self.fill != 0:
            s += ', fill={fill}'
        s += ')'
        d = dict(self.__dict__)
        d['interpolation'] = self.interpolation.value
        return s.format(name=self.__class__.__name__, **d)

def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))
