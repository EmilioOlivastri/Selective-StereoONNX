import sys
sys.path.append("core")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

#from core.raft import RAFTONNX
from core_onnx.raft import RAFTONNX
from utils.utils import InputPadder

DEVICE="cuda"

def replace_instance_norm_with_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.InstanceNorm2d):
            gn = torch.nn.GroupNorm(
                num_groups=child.num_features,
                num_channels=child.num_features,
                affine=child.affine
            )
            setattr(module, name, gn)
        else:
            replace_instance_norm_with_groupnorm(child)
    return module

def load_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]

def export(args):
    model = torch.nn.DataParallel(RAFTONNX(args), device_ids=[0])
    replace_instance_norm_with_groupnorm(model)
    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    model = model.module.eval().to(DEVICE)

    
    image0, image1 = cv2.imread('./onnx_images/left.png'), cv2.imread('./onnx_images/right.png')
    image0 = load_image(cv2.resize(image0, (320, 200), interpolation=cv2.INTER_AREA))
    image1 = load_image(cv2.resize(image1, (320, 200), interpolation=cv2.INTER_AREA))

    padder = InputPadder(image1.shape, divis_by=32)
    image0, image1 = padder.pad(image0, image1)
    images = torch.cat((image0, image1), dim=1).float().to(DEVICE)
    images = images.to(dtype=torch.float32)
    print(f'Image shape: {images.shape}')
    
    print(f'Data Ready! Starting the onnx export...')
    torch.onnx.export(
        model,
        (images,),
        args.output,
        input_names=["images"],
        output_names=["disparity"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"images":    {0: "batch_size"},
                      "disparity": {0: "batch_size"}}
    )
    onnx_model = onnx.load_model(args.output)
    inferred_model = SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=True)
    onnx.save_model(inferred_model, args.output)
    print(f'Successfully exported to: {args.output}')
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", default=None, help="restore checkpoint")
    parser.add_argument("--output", help="Name of output network", default=None)
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--precision_dtype",default="float32",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    parser.add_argument("--valid_iters",type=int,default=32,help="number of flow-field updates during forward pass")

    # Architecture choices
    parser.add_argument("--hidden_dim",nargs="+",type=int,default=128,help="hidden state and context dimensions")
    parser.add_argument("--corr_implementation",choices=["reg", "alt", "reg_cuda", "alt_cuda"],default="reg",help="correlation volume implementation")
    parser.add_argument("--shared_backbone",action="store_true",help="use a single backbone for the context and feature encoders")
    parser.add_argument("--corr_levels",type=int,default=4,help="number of levels in the correlation pyramid")
    parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument("--n_downsample",type=int,default=2,help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--slow_fast_gru",action="store_true",help="iterate the low-res GRUs more frequently")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    export(args)