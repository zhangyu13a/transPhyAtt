import math
import torch.nn.functional as F

def get_params(carlaTcam, carlaTveh):  
    # derived from https://github.com/nlsde-safety-team/DualAttentionAttack/tree/main/src
    scale =0.4                  
    eye = [0, 0, 0]
    for i in range(0, 3):
        # eye[i] = (carlaTcam[0][i] - carlaTveh[0][i]) * scale
        eye[i] = carlaTcam[0][i] * scale 

    # calc camera_direction and camera_up
    pitch = math.radians(carlaTcam[1][0]) 
    yaw = math.radians(carlaTcam[1][1])
    roll = math.radians(carlaTcam[1][2])

    cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    cam_up = [math.cos(math.pi/2+pitch) * math.cos(yaw), math.cos(math.pi/2+pitch) * math.sin(yaw), math.sin(math.pi/2+pitch)]

    p_cam = eye
    p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
    p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
    p_l = [p_cam, p_dir, p_up]
    trans_p = []
    for p in p_l:
        if math.sqrt(p[0]**2 + p[1]**2) == 0:
            cosfi = 0
            sinfi = 0
        else:
            cosfi = p[0] / math.sqrt(p[0]**2 + p[1]**2)
            sinfi = p[1] / math.sqrt(p[0]**2 + p[1]**2)        
        cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
        sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
        trans_p.append([math.sqrt(p[0]**2 + p[1]**2) * cossum, math.sqrt(p[0]**2 + p[1]**2) * sinsum, p[2]])

    return trans_p[0], \
        [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
        [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]