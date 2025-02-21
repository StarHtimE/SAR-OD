## 使用说明
### Step 0  环境准备
```shell
# AutoDL scholar proxy
source /etc/network_turbo
unset http_proxy && unset https_proxy # used to unset proxy

# create env
conda create -y -n DETR python=3.8
conda activate DETR

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# or use pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
### Step 1  安装 mmdet 依赖
```shell
# install dependencies of mmdet
pip install -U openmim
mim install "mmengine==0.8.4"
mim install "mmcv==2.0.1"
```
### Step 2  安装 mmdet
方案 a：如果你开发并直接运行 mmdet，从源码安装它：

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

方案 b：如果你将 mmdet 作为依赖或第三方 Python 包，使用 MIM 安装：

```shell
mim install "mmdet==3.1.0"
```
### Step 3  验证安装
使用示例代码来执行模型推理：

```shell
# download config and checkpoint files
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

**方案 a**：如果你通过源码安装的 MMDetection，那么直接运行以下命令进行验证：

```shell
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

你会在当前文件夹中的 `outputs/vis` 文件夹中看到一个新的图像 `demo.jpg`，图像中包含有网络预测的检测框。

**方案 b**：如果你通过 MIM 安装的 MMDetection，那么可以打开你的 Python 解析器，复制并粘贴以下代码：

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

你将会看到一个包含 `DetDataSample` 的列表，预测结果在 `pred_instance` 里，包含有检测框，类别和得分。

### Step 4  克隆仓库
```shell
git clone https://github.com/StarHtimE/SAR-OD.git
cd SAR-OD
pip install -r requirements.txt
pip install -v -e .
```

### Step 5  准备数据
