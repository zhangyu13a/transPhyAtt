from math import pi
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torchvision.utils import save_image
import imageio
from PIL import Image
from MeshDataset import MeshDataset
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader,
    materials
)
import math
import torch.nn.functional as F

from torchvision.transforms.functional import InterpolationMode
from myRandomAffine import myRandomAffine
from utils import get_params

class MyDataset(Dataset):
    def __init__(self,mesh, data_dir, img_size, device=''):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:           
            self.files.append(file)
        print(len(self.files))
        self.img_size = img_size
        self.device=device
        self.mesh=mesh
        raster_settings = RasterizationSettings(
            image_size= self.img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            # bin_size=0#
            max_faces_per_bin=250000#
        )

        lights = PointLights(device=self.device, location=[[100.0, 85, 100.0]])
        self.cameras=''
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=lights
            )
        )
    
    def set_cameras(self, cameras):
        self.cameras = cameras #
    
    def set_mesh(self, mesh):
        self.mesh = mesh 
  
    def __getitem__(self, index):      
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)
        img = data['img'] 
        veh_trans = data['veh_trans'] 
        cam_trans = data['cam_trans']
        file_name=file.split('/')[-1].split('.npz')[0]       
        scale=150
        for i in range(0, 3):
            cam_trans[0][i] = cam_trans[0][i] * scale 
    
        eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)

        R, T = look_at_view_transform(eye=(tuple(eye),), up=(tuple(camera_up),), at=((0, 0, 0),))
        R[:,:,0]=R[:,:,0]*-1  
        R[:,0,:]=R[:,0,:]*-1
        tmp=R[:,1,:].clone()
        R[:,1,:]=R[:,2,:].clone()
        R[:,2,:]=tmp

        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T,znear=1.0,zfar=300.0,fov=45.0)              
        direction=list(1*np.array(torch.bmm(R,torch.from_numpy(np.array(camera_direction)).unsqueeze(0).unsqueeze(2).float()).squeeze()))
        self.renderer.shader.lights=DirectionalLights(device=self.device, direction=[direction])
        materials = Materials(
                    device=self.device,
                    specular_color=[[1.0, 1.0, 1.0]],
                    shininess=500.0
                )
        self.renderer.rasterizer.cameras=train_cameras
        self.renderer.shader.cameras=train_cameras
        images = self.renderer(self.mesh,materials=materials )      
        imgs_pred = images[:, ..., :3]
       
        img = img[:, :, ::-1]
        img_cv = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img_cv, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0   
        
        bg_shape = img.shape
        car_size=self.renderer.rasterizer.raster_settings.image_size
        dH = bg_shape[2] - car_size
        dW = bg_shape[3] - car_size
        location = (
            dW // 2, 
            dW - (dW // 2), 
            dH // 2, 
            dH - (dH // 2) 
        )
        contour = torch.where((imgs_pred == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))#
        new_contour = torch.zeros(img.permute(0, 2,3, 1).shape, device=self.device)
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_car = torch.zeros(img.permute(0, 2,3, 1).shape, device=self.device)
        new_car[:,:,:,0] = F.pad(imgs_pred[:,:,:,0], location, "constant", value=0)
        new_car[:,:,:,1] = F.pad(imgs_pred[:,:,:,1], location, "constant", value=0)
        new_car[:,:,:,2] = F.pad(imgs_pred[:,:,:,2], location, "constant", value=0)
                         
        total_img = torch.where((new_contour == 0.),img.permute(0, 2,3, 1), new_car)       
        return index,file, total_img.squeeze(0) , imgs_pred.squeeze(0),new_contour.squeeze(0)
    
    def __len__(self):
        return len(self.files)

def initialize_patch(mesh,device,texture_atlas_size):
    print('Initializing patch...')
    sampled_planes = list()  
    with open(r'top_faces_QZH.txt', 'r') as f:
        face_ids = f.readlines() 
        for face_id in face_ids:
            if face_id != '\n':
                sampled_planes.append(int(face_id))              
    idx = torch.Tensor(sampled_planes).long().to(device)
    patch = torch.rand(len(sampled_planes), texture_atlas_size,texture_atlas_size, 3, device=(device), requires_grad=True)#

    return patch,idx


        
    