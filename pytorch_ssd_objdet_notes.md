https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

- [x] Objective
- [x] Concepts
- [x] Overview
- [x] Implementation
- [x] Training
- [x] Evaluation
- [x] Inference
- [x] Frequently Asked Questions

FAQs 里面有一些不错的问题。

- Q: 3x3 kernel 的 prior 经常超出物体边界，为什么？
- A: 3x3 是 kernel size，而不是感受野大小；Keep in mind that the receptive field grows with every successive convolution；For conv_7 and the higher-level feature maps, a 3, 3 kernel's receptive field will cover the entire 300, 300 image. (越深层的特征图，感受野越大)

- Q: While training, why can't we match predicted boxes directly to their ground truths?
- A: 这里想问的其实是为什么要引入 anchor/prior 。减轻训练的难度。predicted boxes are not to be considered reliable, especially during the training process；先让 anchor 和 gt 匹配，然后再用对应的 anchor 去“逼近” gt 。

- Q: Why do we even have a background class if we're only checking which non-background classes meet the threshold?
- A: 为什么要引入背景类？背景类得分高会让其他类别的分数被稀释，更不容易满足 score threshold 的要求，符合直觉。

- Q: Why not simply choose the class with the highest score instead of using a threshold?
- A: 为什么要卡 score threshold 而不是直接用分数最高的类别？目标检测模型的可解释性仍然有待研究，比如，如果一个物体的 bbox 中同时含有大量背景类，怎么办？卡分数阈值也是符合直觉的。

---

## Objective

实现 Single Shot Multibox Detector (SSD)

## Concepts

SSD：
- Two stage 目标检测：
  - RPN 进行 localization
  - classifier 进行分类
- Single-Shot Detection, encapsulate both localization and detection tasks in a single forward sweep of the network，比 two stage 方法的优点是“快”

Multiscale Feature Maps：
- In image classification tasks, we base our predictions on the final convolutional feature map – the smallest but deepest representation of the original image（图像分类，最后的特征图，尺寸最小，最“深”）
- 多尺度特征图，中间的卷积层输出的特征图表达了原始图像不同尺寸上的特征
- 多尺度的作用：检测不同尺寸的物体

Priors:
- Anchor 的概念
-  pre-computed boxes defined at specific positions on specific feature maps, with specific aspect ratios and scales.

Multibox:
- 将预测物体的 bbox 建模成回归问题
- a detected object's coordinates are regressed to its ground truth's coordinates
- for each predicted box, scores are generated for various object types.
（预测的 bbox 是基于 anchor 的）

Hard Negative Mining:
- 选择最难的 false positives （本来是负样本，但被我们预测成正样本）
- 即选择最难预测对的样本
- 对于目标检测任务，如果我们基于 anchor，那么大量的 anchor 都是负样本，使用 Hard Negative Mining 也可以起到正负样本均衡的作用

NMS：
- remove redundant predictions by suppressing all but the one with the maximum score.
- 去重，保留最大分数的检测框

## Overview

这里提到我们会发现 SSD 里有很多工程上的细节，会显得很不直观，很取巧。不要感到困扰。

> Remember, it's built upon years of (often empirical) research in this field.

BBox 的表示
- 可以使用相对于图像宽高的归一化表示。即 (xmin, ymin, xmax, ymax)，都取 0 到 1 的表示。
- 或者用“坐标” + “宽高”，坐标可以用左上角，也可以用中心点

Jaccard Index：IoU，不解释了

Multibox 技术，包含两部分
- box 的坐标，注意 box 里可能不含有任何物体；坐标通过回归得到
- 一个分数向量，每一个分量表示对不同类别的预测分数（置信度）

SSD 的组成：
- VGG Backbone（Base convolutions），利用已有的图像分类基础网络进行特征提取
- Extra Layers（Auxiliary convolutions），高层特征，多尺度
- Multi-box Layers（Prediction convolutions），locate and classify object

SSD300 和 SSD512，数字代表输入图像的尺寸。SSD512 性能更好一些。

### Base Convolutions

为什么需要 backbone ？利用已有的 image classification 模型提供基础特征。

复用已有大数据集的结果。Transfer Learning。

VGG16, 3x3 卷积层的堆叠。

SSD 使用 VGG16 作为 backbone.

As per the paper, we've to make some changes to this pretrained network to adapt it to our own challenge of object detection
(as per: 根据，按照的意思)

作者对原始 VGG16 的一些修改：
- 输入大小改成 300x300 （原来是 224x224）
- 3rd pooling layer 使用 ceiling 代替原来的 floor，目的是保持之后特征图的宽高为偶数
- 5th pooling layer，从 (2,2; stride=2) 改成 (3,3; stride=1)；保持特征图宽高不变（这里应该还得加上 padding=1 才能保持宽高）
- 去掉了全连接层 fc6 和 fc7 ， 用 conv 替代

#### 如何用 conv 替代 FC ？

这里的图很好。

FC 的过程示例：
- 第一步：A 2x2 image with 3 channels is flattened to a 1D vector of size 12.
（把三通道 2x2 图像拉平成一个长度 12 的 1D 向量，12x1）
- 第二步：假设我们希望 FC 之后转为 2 维向量，那么 FC 的 weights 就是 2x12（2x12 * 12x1 = 2x1）

使用 conv 替代 FC：
- 输入还是 3 通道的 2x2 图像
- 使用 kernel size 等于 2x2 的卷积核（注意这里的示例不是用 1x1 conv 代替 FC）
- 两组卷积核，每一组都是 (2x2)x3，这样卷完之后就是 2x1，而且可以证明这里的参数量和上面使用 FC 的参数量是一致的

这里的结论很有用：
- any fully connected layer can be converted to an equivalent convolutional layer simply by reshaping its parameters.

这里怎么用 conv 来替换原始 VGG16 的 fc6 和 fc7？
- fc6 的输入：`7x7x512`，输出：`4096`；我们用 `7x7` kernel size 的卷积核，输出通道数选择 4096 
- （nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=0）
- fc7 的输入：`4096`，输出：`4096`，把输入的 4096 向量想象成有 4096 个通道的 1x1 图像，使用 1x1 卷积核，输出通道数为 4096
- （nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0）

- conv6 has 4096 filters, each with dimensions (7, 7, 512)
- conv7 has 4096 filters, each with dimensions (1, 1, 4096)

这样转换之后的 conv6 和 conv7 还是参数量太大。怎么办？
降采样：
- conv6 will use 1024 filters, each with dimensions 3, 3, 512. Therefore, the parameters are subsampled from 4096, 7, 7, 512 to 1024, 3, 3, 512
- conv7 will use 1024 filters, each with dimensions 1, 1, 1024. Therefore, the parameters are subsampled from 4096, 1, 1, 4096 to 1024, 1, 1, 1024.

这里还引入了 `dilation`，空洞卷积的概念。

修改之后的 base convolutions 的输出特征图尺寸：
- 19x19x1024

### Auxiliary Convolutions

stack some more convolutional layers on top of our base network

关键操作：
- 1x1 卷积层改通道数（宽高不变）
- 3x3 卷积层同时改宽高和通道数（stride 2, pad 1）

feature map 宽高逐次减半。

Priors are manually but carefully chosen based on the shapes and sizes of ground truth objects in our dataset.

在 conv4_3, conv7, conv8_2, conv9_2, conv10_2, and conv11_2 这些不同尺寸的特征图上进行 prior(anchor) 的采样

larger feature maps have priors with smaller scales and are therefore ideal for detecting smaller objects.
- 大尺寸特征图上有更小的尺度，更适合检测小目标

宽高可以用 scale 和 aspect ratio 来表示。
- w x h = s^2
- w/h = a

bbox 回归：归一化，对数表示。

### Prediction convolutions

6个不同尺度的特征图：
- conv4_3, conv7, conv8_2, conv9_2, conv10_2, and conv11_2

What to predict?

for each prior at each location on each feature map：
- the offsets (g_c_x, g_c_y, g_w, g_h) for a bounding box.
- a set of n_classes scores for the bounding box, where n_classes represents the total number of object types (including a background class).
（包含一个背景类）

每个特征图使用两个 convolutional layers
- a localization prediction convolutional layer
- a class prediction convolutional layer

localization prediction convolutional layer
- 3x3 kernel size，pad 取 1 ，stride 取 1 ，每个 prior/anchor 需要预测 4 个值，因此输出通道数为 `4 x num_anchors_per_loc`

class prediction convolutional layer
- 3x3 kernel size，pad 取 1 ，stride 取 1 ，每个 prior/anchor 需要预测 n_classes 个值，因此输出通道数为 `n_classes x num_anchors_per_loc`

localization prediction convolutional layer 和 class prediction convolutional layer 的区别在于通道数

举例：
- FM9_2，即 conv9_2 的输出 feature map
- localization prediction conv layer 对每个点预测 24 个值，即对特征图上每个点预测 6 个 prior/anchor 的 loc offset，所以是 6x24=24 个值
- class prediction conv layer  对每个点预测 `6 x n_classes` 个值

可以对 loc_pred_conv_layer 和 cls_pred_conv_layer 的输出做 reshape，使得它们更符合人类的理解，比如：
- 将 loc_pred reshape 为 (150, 4)，表示 150 个 anchor 的预测结果，每一行 4 个数表示 4 个 loc_offset 的值
- 将 cls_pred reshape 为 (150, 3)，表示 150 个 anchor 的预测结果，每一行 3 个数表示我们要预测的 3 个类别的置信度

### Multibox loss

如何同时计算 regression 和 classification 损失？

如何设计？
- Obviously, our total loss must be an aggregate of losses from both types of predictions – bounding box localizations and class scores.
（我们的 loss 肯定是两种 loss 的“叠加”，如何叠加？）

一些问题？
- What loss function will be used for the regressed bounding boxes?
- Will we use multiclass cross-entropy for the class scores?
- In what ratio will we combine them?
- How do we match predicted boxes to their ground truths?
- We have 8732 predictions! Won't most of these contain no object? Do we even consider them?

#### Matching predictions to ground truths

pred 和 gt 匹配

1. Find the Jaccard overlaps between the 8732 priors and `N` ground truth objects. This will be a tensor of size `(8732, N)` .
2. Match each of the 8732 priors to the object with which it has the greatest overlap.（最优指派问题）
3. 通过 IoU 阈值对 pred 进行过滤，e.g. 小于 0.5 的直接不用；这样把所有 matches 划分为 positive 和 negative

注意，匹配完成后，我们可以对所有 pred 做一个表，每一行是一个 pred，列的属性依次为：
- matched GT
- positive or negative
- label: 要预测的物体类别，e.g. dog, cat, bkgrd
- coordinates: 如果是 positive，给出 gt 的坐标点；换言之，对于 negative match，没有 gt 坐标点，那么在 loss 中如何处理？
- class label：类别标签，positive match 直接用 gt 的 label，negative match 使用背景类

“Localization loss”

localization loss 只对 positive matches 计算。

我们对 (cx cy w h) 编码得到了 (g_cx g_cy g_w g_h)，对 gt 的 bbox 做同样的编码。然后每一个 pred-gt 匹配对计算 Smooth L1 loss 。
（为什么用 smooth L1 loss ？l2对异常点更敏感，smooth l1 loss在|x|>1的部分采用了 l1 loss，避免梯度爆炸。）

“Confidence loss”/"classification loss"

无论是 positive match 还是 negative match 都有一个 type label。（对于 negative match 是 bkgrd）

问题：相比于 positive matches，负样本太多了。
解决方案：控制负样本的数量。How？

Hard Negative Mining：
- only use those predictions where the model found it hardest to recognize that there are no objects. 
（实际上没有物体，但模型说这里有物体，即 false positive）

```
// N_hn: number of hard negatives
// N_p: number of positives
N_hn = 3 * N_p
```

如何找 hard negatives？
- 计算每一个 negative matches 的 Cross Entropy loss，按照 loss 值从大到小排序挑选 N_hn 个

注意这里正负样本都要计算 loss ，但取平均的时候只用 N_{positives}

“Multibox loss”

L_{cls} + \alpha L_{loc}

通过一个比例系数 \alpha 来累加两种 loss 。

SSD 的作者直接使用了 \alpha = 1 。

### Processing predictions

网络输出是两个 tensor，想象成两个矩阵，矩阵行数是 number of anchors。

第一个矩阵是 pred_loc_offsets，第二个矩阵是 pred_cls 。

需要 decoding。

NMS：we suppress all but the one with the maximum score.

还有 confidence score 过滤。

------

## Implementation

数据集，使用 Pascal Visual Object Classes (VOC)，2007，2012

每一个物体的表示：
- bbox
- type label
- difficulty: 0 表示不难，1 表示难

一共 12 个类别。

```
{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
```

模型的输入：
- Images
  - 300x300 大小的输入图像，RGB 格式
  - 使用 VGG-16 base pretrained on ImageNet，像素值需要归一化到 [0, 1]，使用特定的 mean 和 std 进行归一化；
```
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
- Objects' Bounding Boxes
- Objects' Labels

注意对于每一个输入图像，输入到网络里的 tensor shape 是固定的。但 bbox 和 label 的 tensor 形状取决于每一幅图像对应的 gt objects 数量。

### Data pipeline

- 解析原始数据（images, labels）
- PyTorch Dataset, 实现 `__len__` , `__getitem__`；这里也返回了 difficulties，但训练的时候没有用到，只在 eval 时使用了
  - 在 PascalVOCDataset 中也实现了数据增强
    - Randomly adjust brightness, contrast, saturation, and hue，每一种增强有 50% 的概率被选中
    - With a 50% chance, perform a zoom out operation on the image，随机缩小图像，有助于检测小物体，缩小后的边界可以用黑色填充
    - Randomly crop image, i.e. perform a zoom in operation，随机裁剪图像
    - With a 50% chance, horizontally flip the image. 水平翻转图像，不能纵向翻转，这个是先验了，现实世界中左右对称的物体更多，但很少有上下翻转后对称的物体。
    - resize 到 300x300 
    - bbox label 转换为分数表示的坐标（归一化）
    - 使用 ImageNet VGG16 对应的 mean 和 std 对图像进行归一化
- PyTorch DataLoader ，使用 DataLoader 创建用于训练的 batches，设置 `collate_fn` 来进行数据的整理（collate）

接下来就是跟着代码走一遍。

学习这里的代码编写思路：
- 准备数据（utils.py）
- 编写 dataset 和 dataloader 代码（datasets.py）， dataloader 一般在 train 里用到
- 编写数据增强代码（utils.py）
- 搭建网络模型代码（model.py）
  - VGGBase, VGG16 作为 backbone, 基础低层特征提取
  - Auxiliary Convolutions, 辅助卷积层，进一步进行特征抽取，多尺度，高层特征
  - Prediction convolutions, 对不同尺度的特征图进行处理，输出 cls_pred 和 loc_pred
  - SSD300, 将 VGGBase , Auxiliary Convolutions 和 Prediction convolutions 组合在一起成为 SSD 网络
- 编写训练代码（train.py）
- 编写评测代码（eval.py）
  - 评测时候是否要考虑 difficulty？ if the model does detect an object that is considered to be difficult, it must not be counted as a false positive.
- 编写测试代码（detect.py）

### utils.py

```py
import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# 注意这里 label 值从 1 开始
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0  # background 的编号是 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# 这个比较有用，可以用来生成颜色表
# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


# decimate 是抽取的意思，这里是实现类似空洞卷积的降采样
def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
        	# 按照输入的下标 list 进行采样，实现降采样的功能
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


# 计算 mAP 的代码
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


# 边框回归的编码
def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


# 边框回归的解码
def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


# 计算两个集合里 bboxes 的两两相交的面积
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    # 我们要取交集，lower bounds 是最小值里面取最大
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    # upper bounds 是最大值里面取最小
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)  xmax ymax
    # clamp，夹紧
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)  这里最内层一维的 2 个数是 x_diff 和 y_diff
    # element-wise 相乘，即为交集的面积
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


# 计算 IoU 矩阵
def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


# 这里 split 用来标识训练集和测试集，测试集不做数据增强
def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


# 调整学习率，注意这里的设计，参数传入 optimizer
def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


# 计算 topk 准确率
def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


# 保存模型 checkpoint 文件
def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)


# 工具类，计算一些统计指标
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 这里值得注意，控制参数的梯度，防止梯度爆炸；利用了 clamp 方法，“夹紧”，将值限制在一个区间内
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

```

数据集部分的代码：

```py
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        # 注意这里的设计，一个 Dataset 实例根据传入的 split 参数来确定是使用 TRAIN 还是
        # TEST 数据集；而不是我在 OpenPCDet 里实验中做的同时把 train 和 test 加载进来
        assert self.split in {'TRAIN', 'TEST'}

        # 传入 dataset_root_dir
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult  # 是否根据 difficulties 进行过滤

        # 直接读取 json 文件（json 本身就有结构化信息了，这样设计也还可以）
        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')  # 输入需要是 RGB 信息

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
        	# 学习这里的技巧，difficulties 是一个长度为 len(boxes) 的 0 1 数组
        	# 1 - difficulties 会返回一个 0 1 数组，难的 item 对应元素为 0
        	# 可以理解为一个 mask ，boxes = boxes[1 - difficulties] 只保留了难度为 0 ，即简单的 item
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        # 数据增强，同时把 image 转成 tensor
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    # collate 是整理的意思，可以理解为对 batch 中的数据进行调整，整理成我们需要的格式
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        # 这里 images 是 tensor，后面三个是 list ， 每个 list 里面的 tensor 长度不固定
        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

```

模型部分的代码：

`model.py`

```py
from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        # 这里 nn.Conv2d 的第一个参数是 in_channels ， 第二个参数是 out_channels
        # (3, 300, 300) -> (64, 300, 300)  C H W
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        # (64, 300, 300) -> (64, 300, 300)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # (64, 300, 300) -> (64, 150, 150)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 这里会让特征图宽高减半

        # (64, 150, 150) -> (128, 150, 150)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # (128, 150, 150) -> (128, 150, 150)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # (128, 150, 150) -> (128, 75, 75)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (128, 75, 75) -> (256, 75, 75)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # (256, 75, 75) -> (256, 75, 75)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # (256, 75, 75) -> (256, 75, 75)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # (256, 75, 75) -> (256, 38, 38), note the `ceil_mode=True`
        # 使用 MaxPool2d 的 ceil_mode 来处理特征图减半后宽高为奇数的情况
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        # (256, 38, 38) -> (512, 38, 38)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # (512, 38, 38) -> (512, 38, 38)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # (512, 38, 38) -> (512, 38, 38)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # (512, 38, 38) -> (512, 19, 19)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (512, 19, 19) -> (512, 19, 19)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # (512, 19, 19) -> (512, 19, 19)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # (512, 19, 19) -> (512, 19, 19)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # (512, 19, 19) -> (512, 19, 19), 这里的 MaxPool2d 保持了特征图宽高不变
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        # 使用 conv 替代 FC ；注意这里 padding 和 dilation 的选取；dilation=1 相当于不使用 dilation
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        # 1x1 卷积替换 FC，比较直白，输入输出通道数不变
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        # 读取预训练的层
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        # 前馈过程，注意每次 conv 之后都要接上一个 relu
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        # 注意这里用 conv 替换 fc 之后也需要用 relu；“引入非线性”
        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # 返回了两个尺度的特征图
        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and conv7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        # state_dict() 返回的是一个OrderDict，存储了网络中结构的名字和对应的参数
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        # 预训练的 VGG16 的 state_dict，即“参数名 + 参数”
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        # 直接拷贝 conv1 到 conv5 的参数
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # 这里有一些技巧，把预训练 VGG16 的 FC 参数转换成 conv 的参数
        # 先通过 view 进行 reshape, view 的好处是不改变真实的 data，但只能作用于连续数据
        # 然后通过 decimate “抽取”权重，即降采样
        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        # load_state_dict() 是 nn.Module 的方法
        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        # (1024, 19, 19) -> (256, 19, 19), 1x1 conv 改变通道数
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        # (256, 19, 19) -> (512, 10, 10), 3x3 conv, stride=2, padding=1，宽高减半，通道翻倍
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        # (512, 10, 10) -> (128, 10, 10), 1x1 卷积改变通道数
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        # (128, 10, 10) -> (256, 5, 5), 宽高减半，通道翻倍
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        # (256, 5, 5) -> (128, 5, 5), 1x1 卷积改变通道数
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        # (128, 5, 5) -> (256, 3, 3), 宽高减半，通道翻倍
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # (256, 3, 3) -> (128, 3, 3), 1x1 卷积改变通道数
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        # (128, 3, 3) -> (256, 1, 1), 最后的 3x2 卷积相当于得到了一个 256 维的向量
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            # 利用 isinstance 判断是否是卷积层，然后初始化
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.
        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # 返回多尺度的特征图
        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()
        # n_classes 表示类别的数量
        self.n_classes = n_classes

        # 不同特征图上对应的 anchor box 数量
        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # 不同特征图对应的 loc_pred conv layers，注意这里 out_channels 都是 num_anchor_box * 4 ，因为 loc_pred 是预测 4 个 bbox 的偏移量
        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # 不同特征图对应的 cls_pred conv layers，out_channels 是 num_anchor_box * n_classes，预测 n_classes 个类别的置信度
        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.
        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        # NOTE: 这里 permute + contiguous + view 的目的是为了得到最后的 (5776, 4) 形状的 tensor，即 5776 个 1x4 的 bbox offsets
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # 把不同尺度的 pred 结果（loc, cls）concat 在一起
        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # 注意这里的 trick ，先对 conv4_3_feats 进行 L2 归一化，然后乘以 20
        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # 创建 anchor boxes
        # Prior boxes, (8732, 4) tensor
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # conv4_3 特征 L2 归一化，然后乘以 rescale_factor（20）
        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # 提取额外的多尺度特征
        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # 从多尺度特征中获取 loc_pred (8732, 4) 和 cls_pred (8732, n_classes)
        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        # 不同尺度特征图宽高
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        # 不同尺度特征图上 anchor 的 scales
        # FIO: 这些 scale 是怎么设计出来的？
        # 位于网络越深的层, 对应的 anchor 面积越大，所以这里 scales 是递增的
        # 越深的 feature map ，其上面每个像素点的特征值的感受野越大
        # 感受野大小理解为特征图上每个点“捕捉”原始图像信息范围的大小
        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        # 不同尺度特征图上 anchor 的宽高比
        # FIO: 这里会发现 num_scales * num_aspect_ratios 是
        # [3, 5, 5, 5, 3, 3]，而不是我们上面创建 PredictionConvolutions 时的
        # [4, 6, 6, 6, 4, 4]，Why？下面有解释，是对 aspect_ratio 为 1 的时候额外
        # 加了一个 scale
        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):  # 最外层循环是不同特征图
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):  # (i, j) 这里是沿着一个特征图遍历
                    # FIO: 为啥要加 0.5 ？这里坐标都是归一化的
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        # 这里注意宽高和 scale 、 aspect_ratio 的转化关系
                        # w / h = aspect_ratio
                        # w * h = scale^2
                        # w^2 = scale^2 * aspect_ratio
                        # w = scale * sqrt(aspect_ratio)
                        # h = w * h / w
                        #   = (scale^2) / (scale * sqrt(aspect_ratio))
                        #   = scale / sqrt(aspect_ratio)
                        # prior_boxes 里每一行是 [cx cy w h]
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        # 这里的细节，对于宽高比为 1 的情况，额外取一个 scale，其值为
                        # 当前 fmap 和 下一个 fmap 的 scale 的几何平均值
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)  通过 clamp_ 来保证数值范围在 [0, 1]

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        # min_score 是类别置信度阈值
        # max_overlap 是 NMS IoU 阈值
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        # 通过 softmax 得到归一化类别分数；softmax 输入一个向量，输出一个数值归一化到 (0,1) 的向量
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # 先对网络输出的 bbox loc 进行解码
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # 取最大分数和对应的类别 label
            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            # 对每一个类别进行检查
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item() # 下面注释里的 n_qualified 就是这个 n_above_min_score
                # 所有 pred 都小于分数阈值，直接跳过当前帧
                if n_above_min_score == 0:
                    continue
                # 只保留分数大于阈值的 pred
                class_scores = class_scores[score_above_min_score]  # (n_qualified), <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # 学习这里的技巧，通过 sort_ind 对另外一个数组排序
                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_qualified)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_qualified, 4)

                # 计算 IoU 矩阵
                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_qualified)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    # `overlap[box] > max_overlap` 返回的是一个 01 数组
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    # box 自身的 suppress 判断不应该在当前这一次循环里做；注意前面我们
                    # 判断了 suppress[box] == 1 的情况下 continue
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                # `1 - suppress` 是一个 01 数组
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # 没有满足要求的 object ， 用一个 background 类作为占位符
            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        # FIXME: 这里用了 L1Loss，而不是 SmoothL1Loss
        self.smooth_l1 = nn.L1Loss()
        # 使用交叉熵进行分类 loss 计算；通用套路:
        # 1. 神经网络最后一层得到每个类别的得分scores
        # 2. 该得分经过sigmoid(或softmax)函数获得概率输出
        # 3. 模型预测的类别概率输出与真实类别的one hot形式进行交叉熵损失函数的计算。
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        # 和 gt 完成匹配后的 prior 的 gt loc 和 gt class label
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)  # boxes[i] 是 gt boxes

            # gt boxes 和 prior boxes 的 IoU 矩阵
            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # 为每一个 prior box 找 overlap 最大的 gt box
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)  # object_for_each_prior 是一个索引数组，每个元素是 [0, n_objects] 里面的一个数

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior. (有 gt box 没有被匹配到)
            # 2. All priors with the object may be assigned as background based on the threshold (0.5). (由于 score 阈值的原因，关联到某个 gt_box 的 prior boxes 可能都被设置为背景)

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            # 为每一个 gt box 找 overlap 最大的 prior box
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            # 注意 prior_for_each_object 的长度是 n_objects，值是 prior box 的编号，即 [0, 8732-1] 里的一个值
            # 这样能够保证每个 gt 都有对应的 prior
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            # 通过这种方式分配 gt 的 prior，对应的 IoU 直接设置为 1. ，防止被 IoU 阈值过滤掉
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            # 使用 gt 的 class label 给 prior 分配 class label
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            # IoU 小于阈值的 prior 直接设置为背景类
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        # N 是 batch size; 标记每一帧里 priors 是否为 positive
        positive_priors = true_classes != 0  # (N, 8732)

        # 这里暂停一下，计算 MultiBoxLoss 的第一步是“匹配” gt 和 prior

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        # 只对 positive matches 计算 loc loss
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # NOTE: 这里利用了 tensor indexing 的性质，predicted_locs 本来是 (N, 8732)
        #       但我们给的 indexing 数组 positive_priors 跨过了 N & 8732， 结果会有
        #       一个 "flatten" 的效果
        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # NOTE: confidence loss 只选最难的 negative 来计算
        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)  batch 中每一帧图像的 positive priors 数量
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)  batch 中每一帧图像的 hard negative priors 数量

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        # 直接把 postive priors 对应的 loss 设置为 0 ，因为我们下面要根据 loss 
        # 排序，这样的好处是 postive priors 肯定不会出现在前面
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        # expand_as 是扩展为括号内 tensor 一样的形状
        # (1, 8732) -> (N, 8732), 每一行是 [0, 8732-1]
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        # hard_negatives 是一个 01 mask 矩阵，每一行前面的元素为 1 ，用来保留 hard negatives
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # 计算 conf loss，注意分母只用正样本数量
        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        # 通过 self.alpha 控制 conf_loss 和 loc_loss 的比例
        return conf_loss + self.alpha * loc_loss
```

`train.py`

```py
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    # 通过 global 来声明全局变量
    # 这里其实不太好， label_map 是 utils.py 中定义了，上面直接 import * 了
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        # 创建 SSD300 模型
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        # 对 biases 使用两倍的学习率
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    # 注意 .to(device) 的使用
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    # 注意 dataloader 参数
    # shuffle: 每一个epoch进行的时候是否要进行随机打乱
    # num_workers: 使用多少个子进程来同时处理数据加载
    # pin_memory: 如果为True会将数据放置到GPU上去
    # drop_last: 如果最后一点数据如果无法满足batch的大小，那么通过设置True可以将最后部分的数据进行丢弃，否则最后一个batch将会较小。（默认为False）
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # 根据迭代次数、数据集大小和 batch size 来计算 epoch 数量
    # 这里好像有点问题，上面的 batch size 是 8 ，这里沿用了 paper 中的 32 来计算
    epochs = iterations // (len(train_dataset) // 32)
    # decay_lr_at 转换成第几个 epoch 开始 decay
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
        	# 调整学习率，这里的 decay_lr_to 是 scale 系数
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        # 注意传入的参数 ， criterion 这里是 loss 计算的类
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    # 用来统计训练中的各种数据指标
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        # optimizer.zero_grad() 是清空过往梯度信息，pytorch 设计里，不清空的话梯度是会累加的
        # 不调用 zero_grad() 可以实现“梯度累加”技巧：
        # https://www.zhihu.com/question/303070254/answer/573037166
        optimizer.zero_grad()
        loss.backward()

# pytorch 一个训练更新的模板
"""
optimizer.zero_grad()             ## 梯度清零
preds = model(inputs)             ## inference
loss = criterion(preds, targets)  ## 求解loss
loss.backward()                   ## 反向传播求解梯度
optimizer.step()                  ## 更新权重参数

作者：Gary
链接：https://www.zhihu.com/question/303070254/answer/573504133
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""

        # clip 有裁剪、剪切、夹紧的意思，这里是对梯度做限制，防止梯度爆炸
        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # 这里的 loss 是 average loss ，因此传入 images.size(0)，表示这一轮更新的 loss
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        # 打印训练过程中的信息
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    # del 删除的是变量，而不是数据；这里我比较困惑，train 都执行完了为什么还要在这里 del
    # del 做的事情是让数据的引用计数减 1 ，这里相当于手动释放内存了
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
```
