from torch.utils.data import DataLoader, Dataset
import fnmatch
import os
from pytorch3d.io import load_objs_as_meshes, load_obj

class MeshDataset(Dataset):
  def __init__(self, mesh_dir, device, texture_atlas_size=6,shuffle=True, max_num=9999):
    self.len = min(len(fnmatch.filter(os.listdir(mesh_dir), '*.obj')), max_num)
    self.mesh_dir = mesh_dir
    self.shuffle = shuffle

    self.mesh_filenames = fnmatch.filter(os.listdir(mesh_dir), '*.obj')
    self.mesh_filenames = self.mesh_filenames[:self.len]
    self.mesh_files = []
    for m in self.mesh_filenames:
      self.mesh_files.append(os.path.join(self.mesh_dir, m))

    print('Meshes: ', self.mesh_files)
    self.meshes = []
    for mesh in self.mesh_files:
      self.meshes.append(load_objs_as_meshes([mesh], device=device, create_texture_atlas = True,texture_atlas_size = texture_atlas_size))

  def __len__(self):
    return self.len
  
  def __getitem__(self, idx):
    return self.meshes[idx]
