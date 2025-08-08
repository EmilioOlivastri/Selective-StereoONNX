#!/usr/bin/env python3

import argparse
import yaml
import time

import cv2
import numpy as np
import torch

import rospy
import message_filters as mf
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils.utils import InputPadder
from core.raft import RAFT
sys.path.pop()

torch.backends.cudnn.benchmark = True

def create_params_dict(camera_yaml, factor=4.0):
    camera_params = {}
    with open(camera_yaml, 'r') as f:
        stero_params = yaml.safe_load(f)

    inv_res = 1.0 / float(factor) 
    camera_params['fx1'] = float(stero_params['fx1']) * inv_res
    camera_params['fy1'] = float(stero_params['fy1']) * inv_res
    camera_params['cx1'] = float(stero_params['cx1']) * inv_res
    camera_params['cy1'] = float(stero_params['cy1']) * inv_res
    camera_params['cx2'] = float(stero_params['cx2']) * inv_res
    camera_params['baseline'] = float(stero_params['baseline'])

    return camera_params

# Project points in 3D
def unproject(image, disparity, camera_params): 

    fx1, fy1 = camera_params['fx1'], camera_params['fy1']
    cx1, cy1 = camera_params['cx1'], camera_params['cy1']
    cx2, baseline = camera_params['cx2'], camera_params['baseline']

    # Transform disparity in meters
    h, w = disparity.shape
    depth = (fx1 * baseline) / (-disparity + (cx2 - cx1))
    
    # Projection into 3D
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x = (xx - cx1) * depth / fx1
    y = (yy - cy1) * depth / fy1
    z = depth

    # Remove flying points
    mask = np.ones((h, w), dtype=bool)
    mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
    mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False
    
    x, y, z = x[mask], y[mask], z[mask]
    rgb = image[mask]

    # Pack RGB into uint32 using NumPy
    a = np.full((rgb.shape[0],), 255, dtype=np.uint8)
    r, g, b = rgb[:, 2], rgb[:, 1], rgb[:, 0]  # BGR to RGB
    rgba = (a.astype(np.uint32) << 24) | (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    
    # Combine all fields into final Nx4 array
    points = list(zip(x.tolist(), y.tolist(), z.tolist(), rgba.tolist()))

    pointcloud_msg = PointCloud2Msg()
    pointcloud_msg.header.stamp = rospy.Time.now()
    pointcloud_msg.header.frame_id = "camera_frame"

    pointcloud_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
    ]
    pointcloud_msg = point_cloud2.create_cloud(pointcloud_msg.header, pointcloud_msg.fields, points)
    return pointcloud_msg

def load_image(image, device):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
class RosStereoWrapper:
  def __init__(self, args):
    self.args = args
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading stereo model on device: ", self.device)
    self.model = torch.nn.DataParallel(RAFT(args), device_ids=[0])
    self.model.load_state_dict(torch.load(args.restore_ckpt, weights_only=True))
    self.model = self.model.module
    self.model.to(self.device)
    self.model.eval()
    self.resize_factor = 4
    tmp_torch = torch.zeros(1, 3, 200, 320)
    self.padder = InputPadder(tmp_torch.shape, divis_by=32)

    for param in self.model.parameters():
        param.grad = None
    
    self.camera_params = create_params_dict(f'{self.args.camera_params}')

    self.left_sub = mf.Subscriber(f'/stereo/{args.stereo_rig}/left/image_rect', ImageMsg)
    self.right_sub = mf.Subscriber(f'/stereo/{args.stereo_rig}/right/image_rect', ImageMsg)
    
    self.side = args.stereo_rig.split('_')[1]
    self.pointcloud_pub = rospy.Publisher(f'/elas_{self.side}/point_cloud', PointCloud2Msg, queue_size=1)
    self.disp_pub = rospy.Publisher(f'/elas_{self.side}/disparity', ImageMsg, queue_size=1)
    self.ts = mf.ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=1, slop=0.1)
    self.ts.registerCallback(self.run_model)
    print("Stereo model initialized and ready to process images.")


  def imgmsg_to_cv2(self, img_msg):
        
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = []
    if img_msg.encoding == "mono8":
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width), # and one channel of data.
                                    dtype=dtype, buffer=img_msg.data)
    if img_msg.encoding == "bgr8":
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                                dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

  def cv2_to_imgmsg(self, cv_image, encoding="passthrough"):
    img_msg = ImageMsg()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg


  def run_model(self, left_img_msg: ImageMsg, right_img_msg: ImageMsg): 
      """
      Run the stereo model on loaded images and return predictions.
      """

      # Device check
      if not torch.cuda.is_available():
          raise ValueError("CUDA is not available. Check your environment.")

      left_img = self.imgmsg_to_cv2(left_img_msg)
      right_img = self.imgmsg_to_cv2(right_img_msg)

      left_img = cv2.resize(left_img, (left_img.shape[1] // self.resize_factor, left_img.shape[0] // self.resize_factor), interpolation=cv2.INTER_AREA)
      right_img = cv2.resize(right_img, (right_img.shape[1] // self.resize_factor, right_img.shape[0] // self.resize_factor), interpolation=cv2.INTER_AREA)

      # Load images
      left_img_torch = load_image(left_img, self.device)
      right_img_torch = load_image(right_img, self.device)
      
      # Move model to device
      with torch.no_grad():
        left_img_torch, right_img_torch = self.padder.pad(left_img_torch, right_img_torch)
        padding_time = time.time()
        _, flow_up = self.model(left_img_torch, right_img_torch, iters=self.args.valid_iters, test_mode=True)
        inference_time = time.time()
        flow_up = self.padder.unpad(flow_up).squeeze()
      
      print(f"Inference time [{self.side}]: {inference_time - padding_time}")

      np_flowup = flow_up.detach().cpu().numpy().squeeze()  
      del flow_up
      del left_img_torch
      del right_img_torch
      torch.cuda.empty_cache()
      
      disp = -np_flowup.copy().astype(np.int8)
      self.disp_pub.publish(self.cv2_to_imgmsg(disp, encoding="mono8"))

      pt_cloud_msg = unproject(left_img, np_flowup, self.camera_params)
      pt_cloud_msg.header.stamp = left_img_msg.header.stamp
      pt_cloud_msg.header.frame_id = left_img_msg.header.frame_id
      self.pointcloud_pub.publish(pt_cloud_msg)

      return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--camera_params', help="camera parameters file", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument("--precision_dtype",default="float16",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    parser.add_argument('--stereo_rig', type=str, default='stereo_left', help='stereo rig name')


    # Architecture choices
    parser.add_argument("--hidden_dim",nargs="+",type=int,default=128,help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args, unknown = parser.parse_known_args()
    rospy.init_node(f'deep_{args.stereo_rig}_node', anonymous=True)
    rosS_w = RosStereoWrapper(args)
    rospy.spin()


    return

if __name__ == '__main__':
    main()