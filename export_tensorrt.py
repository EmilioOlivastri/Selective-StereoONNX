import sys

sys.path.append("core")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path
import numpy as np
import torch
from utils.utils import InputPadder

import tensorrt as rt

import cv2

from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import (
        CreateConfig,
        EngineFromBytes,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        SaveEngine,
        TrtRunner,
        Profile
    )

DEVICE = "cuda"

def load_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

def export(args):
    
    # Getting the path to the onnx model
    engine_path = Path(args.model_onnx)

    if engine_path.suffix == ".engine":
        build_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
    else:
        profile = Profile()
        profile.add(name="images", 
                    min=(1, 6, 224, 320),   # minimum batch size
                    opt=(2, 6, 224, 320),   # optimal batch size (TensorRT will optimize for this)
                    max=(2, 6, 224, 320))
        tactic_sources = [rt.TacticSource.JIT_CONVOLUTIONS, rt.TacticSource.EDGE_MASK_CONVOLUTIONS, rt.TacticSource.CUBLAS, rt.TacticSource.CUBLAS_LT, rt.TacticSource.CUDNN]
        memory_base = 1024 * 1024 * 1024  # 1GB base memory pool
        memory_pool_limits = {rt.MemoryPoolType.WORKSPACE: int(4 * memory_base)}
        cfg = CreateConfig(fp16=True, memory_pool_limits=memory_pool_limits, 
                           profiles=[profile], tactic_sources=tactic_sources) 
        
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(args.model_onnx)), config=cfg)
        build_engine = SaveEngine(build_engine, str(args.model_onnx) + ".engine")
    
    with TrtRunner(build_engine) as runner:
        image0, image1 = cv2.imread('./onnx_images/left.png'), cv2.imread('./onnx_images/right.png')
        image0 = load_image(cv2.resize(image0, (320, 200), interpolation=cv2.INTER_AREA))
        image1 = load_image(cv2.resize(image1, (320, 200), interpolation=cv2.INTER_AREA))
        
        padder = InputPadder(image1.shape, divis_by=32)
        image0, image1 = padder.pad(image0, image1)
        images = np.concatenate((image0.numpy(), image1.numpy()), axis=1)
        images_dummy = images.copy()
        images_batch = np.concatenate([images_dummy, images_dummy], axis=0)
        for idx in range(10):
            output = runner.infer(feed_dict={"images": images_batch})
            #output = runner.infer(feed_dict={"images": images})
            if idx == 2:
                print('DISPARITY:')
                print(output['disparity'].shape)
        
        print(f'Successfull inference: {runner.last_inference_time():.3f}')


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_onnx", default=None, help="Path to onnx model")
    args = parser.parse_args()

    export(args)