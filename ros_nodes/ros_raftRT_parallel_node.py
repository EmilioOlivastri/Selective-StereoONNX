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
sys.path.pop()

from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

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

def load_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
class RosStereoWrapper:
  def __init__(self, args):
    self.args = args
        
    #### LOADING DEL MODEL ###
    print(f'Loaded model at: {str(args.restore_ckpt)}')
    build_engine = EngineFromBytes(BytesFromPath(str(args.restore_ckpt)))
    self.model = TrtRunner(build_engine)
    self.model.__enter__()

    #### MODEL WARMING-UP #####
    image0, image1 = torch.zeros(1, 3, 200, 320), torch.zeros(1, 3, 200, 320)
    self.padder = InputPadder(image0.shape, divis_by=32)
    image0, image1 = self.padder.pad(image0, image1)
    images = np.concatenate((image0.numpy(), image1.numpy()), axis=1)
    batch = np.concatenate([images, images], axis=0)  # Create a batch of two identical images
    
    print(f'Warming-up initiated!')
    for _ in range(5):
        output = self.model.infer(feed_dict={"images": batch})
    print(f'Warming-up terminated! Last inference speed: {self.model.last_inference_time():.3f} s')

    self.resize_factor = 4  # Resize factor for input images
    self.left_params = create_params_dict(f'{self.args.camera_folder}/stereo_left2_water.yaml')
    self.right_params = create_params_dict(f'{self.args.camera_folder}/stereo_right2_water.yaml')

    self.left_left_sub = mf.Subscriber(f'/stereo/stereo_left/left/image_rect', ImageMsg)
    self.left_right_sub = mf.Subscriber(f'/stereo/stereo_left/right/image_rect', ImageMsg)
    self.right_left_sub = mf.Subscriber(f'/stereo/stereo_right/left/image_rect', ImageMsg)
    self.right_right_sub = mf.Subscriber(f'/stereo/stereo_right/right/image_rect', ImageMsg)
    
    self.pcl_left_pub  = rospy.Publisher(f'/elas_left/point_cloud', PointCloud2Msg, queue_size=1)
    self.pcl_right_pub = rospy.Publisher(f'/elas_right/point_cloud', PointCloud2Msg, queue_size=1)
    self.disp_left_pub = rospy.Publisher(f'/elas_left/disparity', ImageMsg, queue_size=1)
    self.disp_right_pub = rospy.Publisher(f'/elas_right/disparity', ImageMsg, queue_size=1)
    
    self.ts = mf.ApproximateTimeSynchronizer([self.left_left_sub, self.left_right_sub, 
                                              self.right_left_sub, self.right_right_sub], queue_size=1, slop=0.1)
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


  def run_model(self, left_left_img_msg: ImageMsg,  left_right_img_msg: ImageMsg,
                      right_left_img_msg: ImageMsg, right_right_img_msg: ImageMsg):
      """
      Run the stereo model on loaded images and return predictions.
      """

      left_left_img   = self.imgmsg_to_cv2(left_left_img_msg)
      left_right_img  = self.imgmsg_to_cv2(left_right_img_msg)
      right_left_img  = self.imgmsg_to_cv2(right_left_img_msg)
      right_right_img = self.imgmsg_to_cv2(right_right_img_msg)

      left_left_img = cv2.resize(left_left_img, (left_left_img.shape[1] // self.resize_factor, left_left_img.shape[0] // self.resize_factor), interpolation=cv2.INTER_AREA)
      left_right_img = cv2.resize(left_right_img, (left_right_img.shape[1] // self.resize_factor, left_right_img.shape[0] // self.resize_factor), interpolation=cv2.INTER_AREA)
      
      right_left_img = cv2.resize(right_left_img, (right_left_img.shape[1] // self.resize_factor, right_left_img.shape[0] // self.resize_factor), interpolation=cv2.INTER_AREA)
      right_right_img = cv2.resize(right_right_img, (right_right_img.shape[1] // self.resize_factor, right_right_img.shape[0] // self.resize_factor), interpolation=cv2.INTER_AREA)

      # Load images
      left_left_img_torch = load_image(left_left_img)
      left_right_img_torch = load_image(left_right_img)
      right_left_img_torch = load_image(right_left_img)
      right_right_img_torch = load_image(right_right_img)
      
      # Move model to device
      left_left_img_torch, left_right_img_torch = self.padder.pad(left_left_img_torch, left_right_img_torch)
      right_left_img_torch, right_right_img_torch = self.padder.pad(right_left_img_torch, right_right_img_torch)

      input_left = np.concatenate((left_left_img_torch.numpy(), left_right_img_torch.numpy()), axis=1)
      input_right = np.concatenate((right_left_img_torch.numpy(), right_right_img_torch.numpy()), axis=1)
      input_imgs = np.concatenate((input_left, input_right), axis=0)
      
      output = self.model.infer(feed_dict={"images": input_imgs})
      print(f"Inference Time: {self.model.last_inference_time():.3f} s")

      flow_up = torch.from_numpy(output["disparity"])
      flowup_left, flowup_right = flow_up[0, :, :, :], flow_up[1, :, :, :]
      flowup_left = self.padder.unpad(flowup_left[None]).squeeze()
      flowup_right = self.padder.unpad(flowup_right[None]).squeeze()
      np_flowup_left, np_flowup_right = flowup_left.numpy().squeeze(), flowup_right.numpy().squeeze()
      
      del flow_up, flowup_left, flowup_right
      del left_left_img_torch, left_right_img_torch
      del right_left_img_torch, right_right_img_torch
      
      ## STEREO LEFT ##
      disp_left_msg = self.cv2_to_imgmsg(np_flowup_left, encoding="passthrough")
      disp_left_msg.header.frame_id = left_left_img_msg.header.frame_id
      disp_left_msg.header.stamp = left_left_img_msg.header.stamp
      
      ## STEREO RIGHT ##
      disp_right_msg = self.cv2_to_imgmsg(np_flowup_right, encoding="passthrough")
      disp_right_msg.header.frame_id = right_left_img_msg.header.frame_id
      disp_right_msg.header.stamp = right_left_img_msg.header.stamp

      self.disp_left_pub.publish(disp_left_msg)
      self.disp_right_pub.publish(disp_right_msg)

      ## STEREO LEFT ##
      pcl_left_msg = unproject(left_left_img, np_flowup_left, self.left_params)
      pcl_left_msg.header.stamp = left_left_img_msg.header.stamp
      pcl_left_msg.header.frame_id = left_left_img_msg.header.frame_id
      
      ## STEREO RIGHT ##
      pcl_right_msg = unproject(right_left_img, np_flowup_right, self.right_params)
      pcl_right_msg.header.stamp = right_left_img_msg.header.stamp
      pcl_right_msg.header.frame_id = right_left_img_msg.header.frame_id
      
      self.pcl_left_pub.publish(pcl_left_msg)
      self.pcl_right_pub.publish(pcl_right_msg)

      return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--camera_folder', help="Folder with parameters for the cameras", required=True)
    
    args, unknown = parser.parse_known_args()
    rospy.init_node(f'deep_stereo_node', anonymous=True)
    try:
        rosS_w = RosStereoWrapper(args)
        rospy.spin()
    finally:
        rosS_w.model.__exit__(None, None, None)
        print("Model resources released.")


    return

if __name__ == '__main__':
    main()