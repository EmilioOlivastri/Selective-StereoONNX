# Selective-RAFT ONNX

This repo is an adapted version of Selective-Stereo(RAFT) with max compatibility with ONNX and TensorRT exports. This repo is also using a older python version (3.8.10) for compatibility with ROS Noetic. There are some ros nodes provided that, given a topic with rectified stereo matches it provides the dense point cloud. 

This repo has been developed for a NDVIDIA Orin NX with JetPack 5.1.2 - L4T 35.4.1.

The nodes contain both the pytorch version (at the begining I was struggling and couldn't reproduce the same results), and the TensorRT version.

> ✨ ***Original Repo***: The original repo is from [Windsrain](https://github.com/Windsrain/Selective-Stereo). Cite their paper if this repo is helpful and leave a start.

## ⭐ Instruction for installation
The first step to install the repo is to create a conda environment using the environment.yaml file provided
```shell
conda env create -f jetson_env.yaml
conda activate selectiveRaft_jetson
```
For Pytorch a specific version for the Jetson needs to be installed. In my case the version is **2.1.0a0+41361538.nv23.06**. 
Please be sure to check [NVIDIA's support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html). 
Once you have checked the supported version based on your hardware and your Jetpack version (my case 5.1.2), follow this link here:
```
https://developer.download.nvidia.cn/compute/redist/jp/
```
It is going to redirect you to a page with the different version of Jetpack. So click on your version (if my version is 5.1.2 --> click on v512 ), then pytorch and there should be your own cuda enabled pytorch version.
In order to install it in your environemt run the following command (in my case is this) with the env activated:
```shell
pip install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v512/https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl 
```
The original instructions can be found on NVIDIA's [webpage](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Another tricky part of the installation is to find a compliant version of onnxruntime that has gpu. In this [link](https://elinux.org/Jetson_Zoo#ONNX_Runtime) can be found the pip wheel for your python and jetpack version. Dowload it and then run 

```shell
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

The final step is to install a compliant version of TensorRT. In order to do so just follow the instructions provided by NVIDIA [here](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) and follow your preferred installation method.

# ⚠️⚠️⚠️ 
One thing that sometimes happens is that your TensorRT is not found when you create a virtual environment on the Jetson. To make sure that everything is found run the following command while your environemnt is activated.
```shell
export PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH
```
Of course, the command depends on where your python distro is installed.

My TensorRT version = **8.5.2.2**

If there is anything missing in the steps please open and issues!

## ⭐ ONNX Export & Verification
I suggest you to check the original [Stereo-RAFT](https://github.com/princeton-vl/RAFT-Stereo) repo to select which parameters satisfy your requirments for accuracy. The weights that I used were taken from [here](https://github.com/Windsrain/Selective-Stereo). I used the sceneflow.pth.

This is the command that I used to export my model but you can adapt it to your requirmentes:
```shell
python export_onnx.py --restore_ckpt models/sceneflow.pth --valid_iters 10 --shared_backbone --slow_fast_gru --output models/sceneflow.onnx
```
The authors of [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) implement a quick correlation function that is not translatable by ONNX so that possibility has been removed. The components in core_onnx are all ONNX complaiant, at least for OPSET=17. Also in the core_onnx/raft.py some values may be hardcode to avoid weird behaviour from the ONNX export, so just double check that those paramers are actually taking effect. (The number of refinment iterations is fixed to what for me coincided to be realtime for my application!)

To test that the produced network produces similar results to its pytorch version use the following command: 
```shell
python test_onnx_export.py --onnx_model models/sceneflow.onnx --torch_ckpt models/sceneflow.pth --valid_iters 10 --shared_backbone --corr_implementation reg --slow_fast_gru
```
If the max error is over 100 then there is something wrong. In my experience the max < 30 and the mean around 10-15!

## ⭐ TensorRT Export
To make full use of your NVIDIA device's power you can export your ONNX model to TensorRT with the following command:
```shell
python export_tensorrt.py --model_onnx models/sceneflow.onnx
```
Which is, after a while, going to produce a .engine file that then you can use for inference!

## ROS Nodes
Here there are different version of ROS Nodes for running inference on a stream of rectified images. Here are a couple of examples to run Selective-Stereo using Pytorch and TensorRT.

Pytorch:
```shell
python ros_raft_node.py --restore_ckpt ../models/sceneflow.pth --camera_params ../config_stereo/stereo_left.yaml --valid_iters 10 --shared_backbone --corr_implementation reg_cuda --mixed_precision --slow_fast_gru
```

TensorRT:
```shell
python ros_raft_node.py --restore_ckpt models/stereo_realltime.engine --camera_params ../config_stereo/stereo_left.yaml
```

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [Selective-Stereo](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Selective-Stereo_Adaptive_Frequency_Information_Selection_for_Stereo_Matching_CVPR_2024_paper.html) and [Stereo-RAFT](https://ieeexplore.ieee.org/abstract/document/9665883).
```txt
@inproceedings{lipson2021raft,
  title={Raft-stereo: Multilevel recurrent field transforms for stereo matching},
  author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={218--227},
  year={2021},
  organization={IEEE}
}
```

```txt
@inproceedings{wang2024selective,
  title={Selective-stereo: Adaptive frequency information selection for stereo matching},
  author={Wang, Xianqi and Xu, Gangwei and Jia, Hao and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19701--19710},
  year={2024}
}

```