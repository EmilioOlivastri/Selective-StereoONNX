import sys
sys.path.append('core')

import torch
import onnxruntime as ort
import numpy as np
import cv2

from core_onnx.raft import RAFTONNX
from utils.utils import InputPadder


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (320, 200), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # CHW
    return img[None]  # NCHW


def run_pytorch(model, image0, image1):
    padder = InputPadder(image1.shape, divis_by=32)
    image0, image1 = padder.pad(image0, image1)
    images = torch.cat((image0, image1), dim=1).float().cuda()

    with torch.no_grad():
        output = model(images)
    return output.cpu().numpy()


def run_onnx(onnx_path, image0, image1):
    ort_sess = ort.InferenceSession(onnx_path)
    padder = InputPadder(image1.shape, divis_by=32)
    image0, image1 = padder.pad(image0, image1)
    images = torch.cat((image0, image1), dim=1).float()
    ort_inputs = {"images": images.numpy()}
    ort_outs = ort_sess.run(None, ort_inputs)
    return ort_outs[0]


def compare_outputs(torch_output, onnx_output, atol=1e-4, rtol=1e-3):
    if not np.allclose(torch_output, onnx_output, atol=atol, rtol=rtol):
        abs_diff = np.abs(torch_output - onnx_output)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        std_diff = np.std(abs_diff)
        print(f"Outputs differ! Max diff: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Std  absolute difference: {std_diff:.6f}")
    else:
        print("Outputs match within tolerance.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", required=True, help="Path to exported .onnx file")
    parser.add_argument("--torch_ckpt", required=True, help="Path to PyTorch checkpoint")
    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument("--precision_dtype",default="float16",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    

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

    args = parser.parse_args()

    # Load and prepare PyTorch model
    model = RAFTONNX(args)
    model.load_state_dict(torch.load(args.torch_ckpt), strict=False)
    model.eval()
    model.cuda()

    # Load inputs
    image0 = load_image('./onnx_images/left.png')
    image1 = load_image('./onnx_images/right.png')

    torch_out = run_pytorch(model, image0.cuda(), image1.cuda())
    onnx_out = run_onnx(args.onnx_model, image0, image1)

    compare_outputs(torch_out, onnx_out)
