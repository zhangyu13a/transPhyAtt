import os
import torch
from torch.utils.data import DataLoader
from MeshDataset import MeshDataset
from loss import TotalVariation_3d,MaxProbExtractor,NPSCalculator
import time
from torchvision.utils import save_image
from tqdm import tqdm
from YoloV3.yolo_CAM import YOLO
from data_loader_nr import MyDataset


class Patch():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        # Datasets
        self.mesh_dataset = MeshDataset(config.mesh_dir, device, texture_atlas_size=config.texture_atlas_size, max_num=config.num_meshes)
        # Yolo model
        self.dnet =  YOLO(config)
        self.prob_extractor = MaxProbExtractor(cls_id=0, num_cls=1,config=self.config).cuda()

        # Initialize adversarial patch
        self.patch = None
        self.idx = None
        if self.config.patch_dir is not None: 
          self.patch = torch.load(self.config.patch_dir + 'patch_save.pt').to(self.device)
          self.patch.requires_grad=True 
          self.idx = torch.load(self.config.patch_dir + 'idx_save.pt').to(self.device)

        if self.patch is None or self.idx is None:
            self.initialize_patch(device=self.device,texture_atlas_size=config.texture_atlas_size) 
            
        self.nps_calculator = NPSCalculator(self.config.printfile, self.patch.shape).cuda()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
    
    def attack(self):
        mesh = self.mesh_dataset.meshes[0]
        total_variation = TotalVariation_3d(mesh, self.idx).to(self.device)
        optimizer = torch.optim.SGD([self.patch], lr=1e-2, momentum=0.9)
        n_epochs=self.config.epochs

        for epoch in range(n_epochs):          
            for mesh in self.mesh_dataset:

                clamped_patch=self.patch.clone().clamp(min=1e-6, max=0.99999)
                mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch               
                mesh.textures.atlas = mesh.textures._atlas_padded
                mesh.textures._atlas_list = None
                dataset = MyDataset(mesh,self.config.train_dir, self.config.img_size, device=self.device)
                loader = DataLoader(
                    dataset=dataset,     
                    batch_size=self.config.batch_size,  
                    shuffle=self.config.shuffle,  
                    drop_last=self.config.drop_last,                    
                    ) 
                flag=0   
                
                tqdm_loader = tqdm(loader)
                for i, (index,file_name_BS, total_img, texture_img,contour) in enumerate(tqdm_loader):
                    flag=flag+1
                    optimizer.zero_grad()
                    total_img = total_img.permute(0, 3, 1, 2)                                                     #[N H W C]->[N C H W]                                      
                    output=self.dnet.get_output(total_img)
                                       
                    #-----------------------------averaged multi-scale attention map---------------------#
                    inputs = {"image": total_img}
                    self.dnet.multi_attention._register_hook()
                    attention_list,_,_ = self.dnet.multi_attention(inputs,retain_graph=True)                    
                    self.dnet.multi_attention.remove_handlers()
                    #------------------------------------------------------------------------------------#

                    heatmap_constrain_list=list()                                                                  #Foreground Attention
                    heatmap_bg_list=list()                                                                         #Background Attention
                    for kk in range(len(attention_list)):
                        heatmap_constrain_tmp=attention_list[kk]*contour.permute(0,3,1,2) 
                        heatmap_constrain_list.append(heatmap_constrain_tmp)

                        heatmap_bg_tmp=attention_list[kk]-heatmap_constrain_tmp
                        heatmap_bg_list.append(heatmap_bg_tmp)
                        if kk==0:
                            heatmap_constrain=heatmap_constrain_tmp
                            heatmap_bg=heatmap_bg_tmp
                        else:
                            heatmap_constrain=heatmap_constrain+heatmap_constrain_tmp                               #accumulate foreground attention
                            heatmap_bg=heatmap_bg+heatmap_bg_tmp                                                    #accumulate background attention           

                    k_num=100
                    heatmap_constrain2=torch.sum(heatmap_constrain,dim=1).reshape(len(file_name_BS),-1)        
                    heatmap_constrain2_topK=torch.topk(heatmap_constrain2,k=k_num,dim=1,largest=True)                #get the top-K values
                    topK_value,topK_value_indice=heatmap_constrain2_topK[0],heatmap_constrain2_topK[1]
                    heat_average_topk=topK_value.sum(dim=1)/k_num 
                    heat_average=torch.sum(heatmap_constrain,dim=[1,2,3])/torch.sum(heatmap_constrain!=0,dim=[1,2,3])#global average value of foreground attention 
                    heat_loss=torch.mean(heat_average_topk/heat_average)                                             #local average value of foreground attention

                    heatmap_bg2=torch.sum(heatmap_bg,dim=1).reshape(len(file_name_BS),-1)
                    heatmap_bg2_topK=torch.topk(heatmap_bg2,k=k_num,dim=1,largest=True) 
                    topK_value1,topK_value_indice=heatmap_bg2_topK[0],heatmap_bg2_topK[1]
                    heat_average_bg_topk=topK_value1.sum(dim=1)/k_num
                    heat_average_bg=torch.sum(heatmap_bg,dim=[1,2,3])/torch.sum(heatmap_constrain==0,dim=[1,2,3])    #global average value of background attention 
                    heat_loss_bg=torch.mean(heat_average_bg_topk/heat_average_bg)

                    heat_loss=torch.mean(heat_average)*5-torch.mean(heat_average_bg)*5+heat_loss*1-heat_loss_bg*1   #attention loss
                    tv_loss = total_variation(self.patch) * 2.5  
                    nps = self.nps_calculator(self.patch)

                    loss=heat_loss*1+tv_loss*1+nps*1 
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    log_dir=''
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                        tqdm_loader.set_description('Epoch %d/%d ,Loss %.3f,heat_loss_new %.3f,tv_loss %.3f,nps %.3f ' % \
                        (epoch,n_epochs,loss.data.cpu().numpy(),heat_loss.data.cpu().numpy(),tv_loss.data.cpu().numpy(),nps.data.cpu().numpy()))
                        
                        if i==0:                            
                            f.write(time.strftime("%Y%m%d-%H%M%S")+'\n')
                        f.write('Epoch %d/%d ,Loss %.3f,heat_loss_new %.3f,tv_loss %.3f,nps %.3f \n' % \
                        (epoch,n_epochs,loss.data.cpu().numpy(),heat_loss.data.cpu().numpy(),tv_loss.data.cpu().numpy(),nps.data.cpu().numpy()))
                    
                    # Random patch augmentation
                    contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).to(self.device)
                    brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness).to(self.device)
                    noise = torch.FloatTensor(self.patch.shape).uniform_(-1, 1) * self.noise_factor
                    noise = noise.to(self.device)
                    augmented_patch = (self.patch * contrast) + brightness + noise

                    # Clamp patch to avoid PyTorch3D issues
                    clamped_patch = augmented_patch.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
                    mesh.textures.atlas = mesh.textures._atlas_padded
                    mesh.textures._atlas_list = None
                    
                    dataset.set_mesh(mesh)   
                    
                    del  output,total_img, texture_img,contour,  \
                        nps,loss,tv_loss,attention_list
                    torch.cuda.empty_cache()

            patch_save = self.patch.cpu().detach().clone()
            idx_save = self.idx.cpu().detach().clone()
            torch.save(patch_save, 'patch_save.pt')
            torch.save(idx_save, 'idx_save.pt')
           
    def initialize_patch(self,device,texture_atlas_size):
        print('Initializing patch...')
        sampled_planes = list()  
        with open(r'top_faces.txt', 'r') as f:
            face_ids = f.readlines()
            for face_id in face_ids:
                if face_id != '\n':
                    sampled_planes.append(int(face_id))              
        idx = torch.Tensor(sampled_planes).long().to(device)
        patch = torch.rand(len(sampled_planes), texture_atlas_size,texture_atlas_size, 3, device=(device), requires_grad=True)#
        self.idx = idx   #initialize self.idx
        self.patch = patch   #initialize self.patch
    
def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    parser.add_argument('--mesh_dir', type=str, default=r"3d_model")
    parser.add_argument('--patch_dir', type=str, default=None,help='patch_dir is None normally, but it should be a certain path when resuming texture optimization from the last epoch')
    # parser.add_argument('--patch_dir', type=str, default='',help='patch_dir is None normally, but it should be a  certain path when resuming texture optimization from the last epoch')
    # parser.add_argument('--idx', type=str, default='')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=608)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=False,help='whether shuffle the data when training')
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--num_meshes', type=int, default=1)
    parser.add_argument('--texture_atlas_size', type=int, default=1)   
    parser.add_argument('--detector', type=str, default='yolov3')   
    parser.add_argument('--conf_thres', type=int, default=0.25,help='conf_thres of yolov3')
    parser.add_argument('--iou_thres', type=int, default=0.5,help='iou_thres of yolov3')
    parser.add_argument('--printfile', type=str, default=r'non_printability\30values.txt')
    parser.add_argument('--train_dir', type=str, default=r'')
    parser.add_argument('--weightfile', type=str, default=r"")    
    config = parser.parse_args()
    trainer = Patch(config, device)
    
    if config.detector == 'yolov3':
        trainer.attack() 
if __name__ == '__main__':
    main()
