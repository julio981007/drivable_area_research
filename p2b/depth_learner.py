import numpy as np
from params import Params
import sys

class inv_proj(object):
    def __init__(self, img_width, img_height):
        self.w = img_width
        self.h = img_height
        self.c = 3
        self.translation_vector_b = np.array([[Params['b_x']], [Params['b_y']], [Params['b_z']]])
        self.translation_vector_t = np.array([[Params['t_x']], [Params['t_y']], [Params['t_z']]])
        
        self.P_ = Params['P'][:, :3]
        self.R0 = Params['R0']
        self.Rt = Params['Rt']
        self.R_ = Params['Rt'][:, :3]
        
    def get_3dpoint_from_depthmap(self, depth_image):
        imageplane_depth = self.get_pixel_depth(depth_image)
        after_P__depth = self.inverse_projection_P(imageplane_depth)
        after_R0_depth = self.inverse_projection_R0(after_P__depth)
        after_Rt_depth = self.inverse_projection_Rt(after_R0_depth)
        depth_point_3d = after_Rt_depth.T
        return depth_point_3d
        
    def get_pixel_depth(self, depth_image):
        depth_values = depth_image[:,:]
        
        nonzero_indices = np.nonzero(depth_values)
        coordinates_values = []
        for i in range(len(nonzero_indices[0])):
            x = nonzero_indices[0][i]
            y = nonzero_indices[1][i]
            value = depth_values[x, y]
            coordinates_values.append([x, y, value])
            
        pixel_depth = np.array(coordinates_values)
        pixel_depth = pixel_depth.astype(float)
        imageplane_depth = np.zeros(pixel_depth.shape)
        imageplane_depth[:, 0] = pixel_depth[:, 1] * pixel_depth[:, 2]
        imageplane_depth[:, 1] = pixel_depth[:, 0] * pixel_depth[:, 2]
        imageplane_depth[:, 2] = pixel_depth[:, 2]
        imageplane_depth_T = imageplane_depth.T
        
        return imageplane_depth_T
    
    def inverse_projection_P(self, imageplane_depth):
        after_b_tr_depth = imageplane_depth - self.translation_vector_b
        after_P_depth = np.dot(np.linalg.inv(self.P_), after_b_tr_depth)
        return after_P_depth
    
    def inverse_projection_R0(self, after_P__depth):
        after_R0_depth = np.dot(np.linalg.inv(self.R0), after_P__depth)
        return after_R0_depth
    
    def inverse_projection_Rt(self, after_R0_depth):
        after_t_depth = after_R0_depth - self.translation_vector_t
        after_Rt_depth = np.dot(np.linalg.inv(self.R_), after_t_depth)
        return after_Rt_depth