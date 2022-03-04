# 2021-Paddle-Temporal-Action-Proposal-TOP-1
# 基于飞桨实现乒乓球时序动作定位大赛: B榜第一名方案

# Code page: https://aistudio.baidu.com/aistudio/projectdetail/3545680?shared=1

# 一、赛题介绍
时序动作定位(提案生成)是计算机视觉和视频分析领域一个具有的挑战性的任务。本次比赛不同于以往的ActivityNet-TAL，FineAction等视频时序检测动作定位比赛，我们采用了更精细的动作数据集--乒乓球转播画面，该数据集具有动作时间跨度短，分布密集等特点，给传统模型精确定位细粒度动作带来了很大挑战。本次比赛的任务即针对乒乓球转播画面视频面对镜头的运动员定位其挥拍动作(时序动作提案生成)。

# 二、竞赛数据集
数据集包含了$19-21$赛季兵乓球国际（世界杯、世锦赛、亚锦赛，奥运会）国内（全运会，乒超联赛）比赛标准单机位高清转播画面特征信息。其中包含$912$条视频特征文件，每个视频时长在$0～6$分钟不等，特征维度为$2048$，以pkl格式保存。我们对特征数据中面朝镜头的运动员的回合内挥拍动作进行了标注，单个动作时常在$0～2$秒不等，训练数据为$729$条标注视频，A测数据为$91$条视频，B测数据为$92$条视频，训练数据标签以json格式给出

- 训练数据集与测试数据集的目录结构如下所示：
```python
| - data
	| - data123004
		| - Features_competition_test_A.tar.gz
	| - data122998
		| - Features_competition_train.tar.gz
		| - label_cls14_train.json 
```
- 本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集(第二阶段公布)三个部分，其中训练集共$729$个样本(视频)，A榜测试集共$91$个样本，B榜测试集共$92$个样本；
- Features目录中包含$912$条ppTSM抽取的视频特征，特征保存为pkl格式，文件名对应视频名称，读取pkl之后以(num_of_frames, $2048$)向量形式代表单个视频特征，如下示例
```python
{'image_feature': array([[-0.00178786, -0.00247065,  0.00754537, ..., -0.00248864,
        -0.00233971,  0.00536158],
       [-0.00212389, -0.00323782,  0.0198264 , ...,  0.00029546,
        -0.00265382,  0.01696528],
       [-0.00230571, -0.00363361,  0.01017699, ...,  0.00989012,
        -0.00283369,  0.01878656],
       ...,
       [-0.00126995,  0.01113492, -0.00036558, ...,  0.00343453,
        -0.00191288, -0.00117079],
       [-0.00129959,  0.01329842,  0.00051888, ...,  0.01843636,
        -0.00191984, -0.00067066],
       [-0.00134973,  0.02784026, -0.00212213, ...,  0.05027904,
        -0.00198008, -0.00054018]], dtype=float32)}
<class 'numpy.ndarray'>
```
- 训练标签见如下格式：
```javascript
# label_cls14_train.json
{
    'fps': 25,    #视频帧率
    'gts': [
        {
            'url': 'name_of_clip.mp4',      #名称
            'total_frames': 6341,    #总帧数（这里总帧数不代表实际视频帧数，以特征数据集维度信息为准）
            'actions': [
                {
                    "label_ids": [7],    #动作类型编号
                    "label_names": ["name_of_action"],     #动作类型
                    "start_id": 201,  #动作起始时间,单位为秒
                    "end_id": 111    #动作结束时间,单位为秒
                },
                ...
            ]
        },
        ...
    ]
}
```

# 三、模型构建思路及调优过程

## 1、数据预处理与数据增强

数据预处理方面，为了实现数据均衡，我们尽量使每个动作片段包含的动作数相当，并且保证不能切断任何动作（参考PaddleVideo足球视频动作提取的预处理）：

- 删去没有动作的片段；
- 片段内如出现多动作，则将其拆分并且重叠提取。

数据增强方面，我们在训练过程中添加加性噪声$a$与乘性噪声$b$，并以0.5的概率对任一片段使用（$a\sim\mathcal{N}(0,0.05)$，$b\sim\mathcal{N}(1,0.05)$）。

$$
x =
\begin{cases}
b\times x + a, & selected  \\
x, & others
\end{cases}
$$

## 2、模型

BMN模型作为本次比赛的baseline，是$2019$年ActivityNet夺冠方案，为视频动作定位问题中proposal的生成提供高效的解决方案。此模型引入边界匹配(Boundary-Matching, BM)机制来评估proposal的置信度，按照proposal开始边界的位置及其长度将所有可能存在的proposal组合成一个二维的BM置信度图，图中每个点的数值代表其所对应的proposal的置信度分数。网络由三个模块组成，Base Module(BM)作为主干网络处理输入的特征序列，TEM模块预测每一个时序位置属于动作开始、动作结束的概率，PEM模块生成BM置信度图。在对BMN适当调参后，全料数据训练并且做完交叉验证在A榜获得了$47$的分数。我们依然沿用BMN网络的三模块结构，并且参照CVPR2020-HACS挑战赛中的时序动作检测任务冠军算法TCANet，对BMN网络做了稍微的修改。具体地，我们在BM模块后增加了局部-全局时序特征编码器(LGTE)和特征注意力模块(SEBlock)。修改后的网络命名为Attention-guided Boundary-Matching Network(ABMN).

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/8f09b5d1c30a4fff9fdb6a94e09f7676c0c7aa9199fc46caaf86ee6c96706d87" width="1000" height="900">
</p>

具体模型设计可参考原论文:

[BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.pdf), Lin et al., Baidu Inc.

[Temporal Context Aggregation Network for Temporal Action
Proposal Refinement](https://openaccess.thecvf.com/content/CVPR2021/papers/Qing_Temporal_Context_Aggregation_Network_for_Temporal_Action_Proposal_Refinement_CVPR_2021_paper.pdf), Lin et al., Baidu Inc.

## 3、模型集成

我们将BMN模型和ABMN模型进行集成，集成方式是将每个模型预测得到的置信得分相乘，最后的置信分是相乘之后的得分。

$$S = S_{BMN}\times S_{ABMN}$$

## 4、参数设置

两个模型参数设置相同，$tscale$为$300$时效果最优，由于每个动作不超过$2$秒，帧率$25$，所以$dscale$设置为$50$时最优。

```bibtex
MODEL: #MODEL field
    framework: "BMNLocalizer"
    backbone:
        name: "BMN" # or "ABMN"
        tscale: 300
        dscale: 50
        prop_boundary_ratio: 0.5
        num_sample: 32
        num_sample_perbin: 3
        feat_dim: 2048
    loss:
        name: "BMNLoss"
        tscale: 300
        dscale: 50

DATASET: #DATASET field
    batch_size: 4 #single card bacth size
    test_batch_size: 1
    num_workers: 8
    train:
        format: "BMNDataset"
        file_path: "/home/aistudio/data/new_label_cls14_train.json"
        subset: "train"
    valid:
        format: "BMNDataset"
        file_path: "/home/aistudio/data/new_label_cls14_train.json"
        subset: "validation"
    test:
        format: "BMNDataset"
        test_mode: True
        file_path: "/home/aistudio/data/new_label_cls14_train.json"
        subset: "validation"

PIPELINE: #PIPELINE field
    train: #Mandotary, indicate the pipeline to deal with the training data, associate to the 'paddlevideo/loader/pipelines/'
        load_feat:
            name: "LoadFeat"
            feat_path: "/home/aistudio/data/Features_competition_train/npy"
        transform: #Mandotary, image transfrom operator
        - GetMatchMap:
            tscale: 300
        - GetVideoLabel:
            tscale: 300
            dscale: 50

    valid: #Mandotary, indicate the pipeline to deal with the training data, associate to the 'paddlevideo/loader/pipelines/'
        load_feat:
            name: "LoadFeat"
            feat_path: "/home/aistudio/data/Features_competition_train/npy"
        transform: #Mandotary, image transfrom operator
        - GetMatchMap:
            tscale: 300
        - GetVideoLabel:
            tscale: 300
            dscale: 50

    test: #Mandatory, indicate the pipeline to deal with the validing data. associate to the 'paddlevideo/loader/pipelines/'
        load_feat:
            name: "LoadFeat"
            feat_path: "/home/aistudio/data/Features_competition_train/npy"
        transform: #Mandotary, image transfrom operator
        - GetMatchMap:
            tscale: 300
        - GetVideoLabel:
            tscale: 300
            dscale: 50

OPTIMIZER: #OPTIMIZER field
    name: 'Adam'
    learning_rate:
        iter_step: True
        name: 'CustomPiecewiseDecay'
        boundaries: [32800]
        values: [0.0008, 0.00008]
    weight_decay:
        name: 'L2'
        value: 1e-4

# OPTIMIZER: #OPTIMIZER field
#     name: 'Adam'
#     learning_rate:
#         iter_step: True
#         name: 'CustomPiecewiseDecay'
#         boundaries: [13800]
#         values: [0.00008, 0.00004]
#     weight_decay:
#         name: 'L2'
#         value: 1e-4

# OPTIMIZER: #OPTIMIZER field
#     name: 'Adam'
#     learning_rate:
#         iter_step: True
#         name: 'CustomWarmupCosineDecay'
#         max_epoch: 5
#         warmup_epochs: 1
#         warmup_start_lr: 0.00006
#         cosine_base_lr: 0.00008
#     weight_decay:
#         name: 'L2'
#         value: 1e-4

METRIC:
    name: 'BMNMetric'
    tscale: 300
    dscale: 50
    file_path: "data/bmn_data/activitynet_1.3_annotations.json"
    ground_truth_filename: "data/bmn_data/activity_net_1_3_new.json"
    subset: "validation"
    output_path: "data/bmn/BMN_Test_output"
    result_path: "data/bmn/BMN_Test_results"

INFERENCE:
    name: 'BMN_Inference_helper'
    feat_dim: 2048
    dscale: 50
    tscale: 300
    result_path: "data/bmn/BMN_INFERENCE_results"

model_name: BMN  # or "ABMN"
epochs: 13 #Mandatory, total epoch
log_level: "INFO"
resume_from: "" #checkpoint path.
```

## 5、调参实验记录

| model     | learning rate | window size | tscale | dsacle | preprocess | data augumentation | fine tune | TTA | results |
| ----------- | ----------- |----------- |----------- |----------- |----------- |----------- |----------- |----------- |----------- |
| BMN      |  CWCD |$8$       |$300$ |$300$ | $\times$ | $\times$ |$\times$  |$\times$  |$42.86$ |
| BMN      |  CWCD |$18$       |$300$ |$300$ |$\times$ | $\times$ |$\times$  |$\times$  |$43.11$ |
| BMN      |  CWCD |$12$       |$300$ |$300$ | $\times$ | $\times$ | $\times$ |$\times$  |$43.25$ |
| BMN      |  CWCD|$12$       |$300$ |$300$ | $\times$ | $\times$ |$\surd$ |$\times$  |$44.10$ |
| BMN      |  CWCD |$12$       |$300$ |$300$ | $\times$ |$\surd$ | $\times$ |$\times$  |$43.64$ |
| BMN      |  CWCD |$12$       |$300$ |$300$ | $\times$ |$\surd$ |$\surd$ |$\times$  |$43.74$ |
| BMN      |  CWCD |$12$      |$300$ |$300$ |$\surd$ |$\surd$ | $\times$ |$\times$  |$44.93$ |
| BMN      |  CWCD |$12$      |$300$ |$300$ |$\surd$ |$\surd$ |$\surd$ |$\times$  |$45.76$ |
| BMN      |  CWCD |$12$       |$300$ |$100$ |$\surd$ |$\surd$ |$\surd$ |$\times$  |$46.03$ |
| BMN      |  CWCD |$12$       |$300$ |$60$  |$\surd$ |$\surd$ |$\surd$ |$\times$  |$46.69$ |
| BMN      |  CWCD |$12$      |$300$ |$50$  |$\surd$ |$\surd$ |$\surd$ |$\times$  |$46.71$ |
| BMN      |  CWCD|$12$       |$300$ |$50$  |$\surd$ |$\surd$ |$\surd$ |$\surd$  |$46.74$ |
| BMN      |  CPWD|$12$      |$300$ |$50$  |$\surd$ |$\surd$ |$\surd$ |$\times$  |$46.96$ |
| ABMN     |  CPWD |$12$       |$300$ |$50$  |$\surd$ |$\surd$ |$\surd$ |$\times$  |$47.50$ |
| BMN+ABMN (add)     |  CPWD |$12$      |$300$ |$50$  |$\surd$ |$\surd$ |$\surd$ |$\times$  |$47.52$ |
| BMN+ABMN (multiply)     |  CPWD |$12$      |$300$ |$50$  |$\surd$ |$\surd$ |$\surd$ |$\times$  |$47.77$ |

CWCD：CustomWarmupCosineDecay
CPWD：CustomPiecewiseDecay

TTA无明显提高且增加推理时间，故决定放弃。

ABMN相比于BMN，在训练迭代中更稳定，相邻epoch的模型得分在A榜相差不大，且不容易过拟合。

## 6、Baseline Bug修改

- `work/paddlevideo/loader/pipelines/anet_pipeline.py`第$50$行$tscale$修改为$dscale$，否则$tscale$与$dscale$不相等时报错；
- 在`work/tools/utils.py`的`score_vector_list.append([xmin, xmax, conf_score])`后面加上以下代码，否则片段无动作提名时报错：

```
if score_vector_list == []:
	return 0
```

# 四、模型训练（以下代码直接在终端输入即可，如果想复现我们的结果，请跳过2-6，执行完1后直接执行7-11）


## 1、环境配置

```
cd work/script
sh environment.sh
```

## 2、解压训练集

```
sh unzip_tra_dataset.sh
```

## 3、训练集预处理

```
sh tra_preprocess.sh
```

## 4、k折划分（可选）

```
sh split_k_fold.sh
```

## 5、训练（如果选择了4，需要相应修改yaml文件中的数据路径）

```
sh train.sh
```

你会看到如下信息：

```
[03/03 20:30:52] DALI is not installed, you can improve performance if use DALI
[03/03 20:30:52] DATASET : 
[03/03 20:30:52]     batch_size : 4
[03/03 20:30:52]     num_workers : 8
[03/03 20:30:52]     test : 
[03/03 20:30:52]         file_path : /home/aistudio/data/new_label_cls14_train.json
[03/03 20:30:52]         format : BMNDataset
[03/03 20:30:52]         subset : validation
[03/03 20:30:52]         test_mode : True
[03/03 20:30:52]     test_batch_size : 1
[03/03 20:30:52]     train : 
[03/03 20:30:52]         file_path : /home/aistudio/data/new_label_cls14_train.json
[03/03 20:30:52]         format : BMNDataset
[03/03 20:30:52]         subset : train
[03/03 20:30:52]     valid : 
[03/03 20:30:52]         file_path : /home/aistudio/data/new_label_cls14_train.json
[03/03 20:30:52]         format : BMNDataset
[03/03 20:30:52]         subset : validation
[03/03 20:30:52] ------------------------------------------------------------
[03/03 20:30:52] INFERENCE : 
[03/03 20:30:52]     dscale : 50
[03/03 20:30:52]     feat_dim : 2048
[03/03 20:30:52]     name : BMN_Inference_helper
[03/03 20:30:52]     result_path : data/bmn/BMN_INFERENCE_results
[03/03 20:30:52]     tscale : 300
[03/03 20:30:52] ------------------------------------------------------------
[03/03 20:30:52] METRIC : 
[03/03 20:30:52]     dscale : 50
[03/03 20:30:52]     file_path : data/bmn_data/activitynet_1.3_annotations.json
[03/03 20:30:52]     ground_truth_filename : data/bmn_data/activity_net_1_3_new.json
[03/03 20:30:52]     name : BMNMetric
[03/03 20:30:52]     output_path : data/bmn/BMN_Test_output
[03/03 20:30:52]     result_path : data/bmn/BMN_Test_results
[03/03 20:30:52]     subset : validation
[03/03 20:30:52]     tscale : 300
[03/03 20:30:52] ------------------------------------------------------------
[03/03 20:30:52] MODEL : 
[03/03 20:30:52]     backbone : 
[03/03 20:30:52]         dscale : 50
[03/03 20:30:52]         feat_dim : 2048
[03/03 20:30:52]         name : BMN
[03/03 20:30:52]         num_sample : 32
[03/03 20:30:52]         num_sample_perbin : 3
[03/03 20:30:52]         prop_boundary_ratio : 0.5
[03/03 20:30:52]         tscale : 300
[03/03 20:30:52]     framework : BMNLocalizer
[03/03 20:30:52]     loss : 
[03/03 20:30:52]         dscale : 50
[03/03 20:30:52]         name : BMNLoss
[03/03 20:30:52]         tscale : 300
[03/03 20:30:52] ------------------------------------------------------------
[03/03 20:30:52] OPTIMIZER : 
[03/03 20:30:52]     learning_rate : 
[03/03 20:30:52]         boundaries : [32800]
[03/03 20:30:52]         iter_step : True
[03/03 20:30:52]         name : CustomPiecewiseDecay
[03/03 20:30:52]         values : [0.0008, 8e-05]
[03/03 20:30:52]     name : Adam
[03/03 20:30:52]     weight_decay : 
[03/03 20:30:52]         name : L2
[03/03 20:30:52]         value : 0.0001
[03/03 20:30:52] ------------------------------------------------------------
[03/03 20:30:52] PIPELINE : 
[03/03 20:30:52]     test : 
[03/03 20:30:52]         load_feat : 
[03/03 20:30:52]             feat_path : /home/aistudio/data/Features_competition_train/npy
[03/03 20:30:52]             name : LoadFeat
[03/03 20:30:52]         transform : 
[03/03 20:30:52]             GetMatchMap : 
[03/03 20:30:52]                 tscale : 300
[03/03 20:30:52]             GetVideoLabel : 
[03/03 20:30:52]                 dscale : 50
[03/03 20:30:52]                 tscale : 300
[03/03 20:30:52]     train : 
[03/03 20:30:52]         load_feat : 
[03/03 20:30:52]             feat_path : /home/aistudio/data/Features_competition_train/npy
[03/03 20:30:52]             name : LoadFeat
[03/03 20:30:52]         transform : 
[03/03 20:30:52]             GetMatchMap : 
[03/03 20:30:52]                 tscale : 300
[03/03 20:30:52]             GetVideoLabel : 
[03/03 20:30:52]                 dscale : 50
[03/03 20:30:52]                 tscale : 300
[03/03 20:30:52]     valid : 
[03/03 20:30:52]         load_feat : 
[03/03 20:30:52]             feat_path : /home/aistudio/data/Features_competition_train/npy
[03/03 20:30:52]             name : LoadFeat
[03/03 20:30:52]         transform : 
[03/03 20:30:52]             GetMatchMap : 
[03/03 20:30:52]                 tscale : 300
[03/03 20:30:52]             GetVideoLabel : 
[03/03 20:30:52]                 dscale : 50
[03/03 20:30:52]                 tscale : 300
[03/03 20:30:52] ------------------------------------------------------------
[03/03 20:30:52] epochs : 13
[03/03 20:30:52] log_level : INFO
[03/03 20:30:52] model_name : BMN
[03/03 20:30:52] resume_from : 
W0303 20:30:52.452201  4531 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0303 20:30:52.457406  4531 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[03/03 20:31:11] train subset video numbers: 18526
[03/03 20:31:13] epoch:[  1/13 ] train step:0    loss: 2.61718 lr: 0.000800 batch_cost: 2.14211 sec, reader_cost: 1.95463 sec, ips: 1.86731 instance/sec.
[03/03 20:31:15] epoch:[  1/13 ] train step:10   loss: 1.99536 lr: 0.000800 batch_cost: 0.14552 sec, reader_cost: 0.00178 sec, ips: 27.48818 instance/sec.
[03/03 20:31:17] epoch:[  1/13 ] train step:20   loss: 1.77924 lr: 0.000800 batch_cost: 0.15357 sec, reader_cost: 0.00034 sec, ips: 26.04741 instance/sec.
[03/03 20:31:18] epoch:[  1/13 ] train step:30   loss: 1.50456 lr: 0.000800 batch_cost: 0.14595 sec, reader_cost: 0.00038 sec, ips: 27.40717 instance/sec.
[03/03 20:31:20] epoch:[  1/13 ] train step:40   loss: 1.22430 lr: 0.000800 batch_cost: 0.20066 sec, reader_cost: 0.00035 sec, ips: 19.93382 instance/sec.
[03/03 20:31:22] epoch:[  1/13 ] train step:50   loss: 1.43952 lr: 0.000800 batch_cost: 0.20048 sec, reader_cost: 0.00036 sec, ips: 19.95176 instance/sec.
[03/03 20:31:24] epoch:[  1/13 ] train step:60   loss: 1.51576 lr: 0.000800 batch_cost: 0.15292 sec, reader_cost: 0.00174 sec, ips: 26.15751 instance/sec.
[03/03 20:31:25] epoch:[  1/13 ] train step:70   loss: 1.28944 lr: 0.000800 batch_cost: 0.14365 sec, reader_cost: 0.00029 sec, ips: 27.84559 instance/sec.
[03/03 20:31:27] epoch:[  1/13 ] train step:80   loss: 1.35131 lr: 0.000800 batch_cost: 0.15031 sec, reader_cost: 0.00036 sec, ips: 26.61238 instance/sec.
[03/03 20:31:28] epoch:[  1/13 ] train step:90   loss: 1.09097 lr: 0.000800 batch_cost: 0.15946 sec, reader_cost: 0.00171 sec, ips: 25.08405 instance/sec.
[03/03 20:31:30] epoch:[  1/13 ] train step:100  loss: 1.33501 lr: 0.000800 batch_cost: 0.16283 sec, reader_cost: 0.00039 sec, ips: 24.56548 instance/sec.
[03/03 20:31:31] epoch:[  1/13 ] train step:110  loss: 1.39534 lr: 0.000800 batch_cost: 0.14575 sec, reader_cost: 0.00036 sec, ips: 27.44465 instance/sec.
[03/03 20:31:33] epoch:[  1/13 ] train step:120  loss: 1.08748 lr: 0.000800 batch_cost: 0.14481 sec, reader_cost: 0.00063 sec, ips: 27.62245 instance/sec.
[03/03 20:31:34] epoch:[  1/13 ] train step:130  loss: 1.46802 lr: 0.000800 batch_cost: 0.14575 sec, reader_cost: 0.00039 sec, ips: 27.44438 instance/sec.
[03/03 20:31:36] epoch:[  1/13 ] train step:140  loss: 1.55028 lr: 0.000800 batch_cost: 0.15472 sec, reader_cost: 0.00182 sec, ips: 25.85386 instance/sec.
[03/03 20:31:37] epoch:[  1/13 ] train step:150  loss: 1.16257 lr: 0.000800 batch_cost: 0.14537 sec, reader_cost: 0.00035 sec, ips: 27.51572 instance/sec.
[03/03 20:31:39] epoch:[  1/13 ] train step:160  loss: 0.98415 lr: 0.000800 batch_cost: 0.14494 sec, reader_cost: 0.00038 sec, ips: 27.59692 instance/sec.
[03/03 20:31:40] epoch:[  1/13 ] train step:170  loss: 1.09375 lr: 0.000800 batch_cost: 0.14334 sec, reader_cost: 0.00041 sec, ips: 27.90510 instance/sec.
[03/03 20:31:42] epoch:[  1/13 ] train step:180  loss: 0.98302 lr: 0.000800 batch_cost: 0.15417 sec, reader_cost: 0.00030 sec, ips: 25.94566 instance/sec.
[03/03 20:31:43] epoch:[  1/13 ] train step:190  loss: 0.84935 lr: 0.000800 batch_cost: 0.14572 sec, reader_cost: 0.00054 sec, ips: 27.44900 instance/sec.
[03/03 20:31:45] epoch:[  1/13 ] train step:200  loss: 0.73159 lr: 0.000800 batch_cost: 0.15281 sec, reader_cost: 0.00038 sec, ips: 26.17624 instance/sec.
[03/03 20:31:46] epoch:[  1/13 ] train step:210  loss: 0.78628 lr: 0.000800 batch_cost: 0.15997 sec, reader_cost: 0.00034 sec, ips: 25.00479 instance/sec.
```

## 6、提取模型

```
sh export_model.sh
```

## 7、解压模型

```
sh unzip_model.sh
```

## 8、解压验证集

```
sh unzip_val_dataset.sh
```

## 9、验证集预处理

```
sh val_preprocess.sh
```

## 10、推理

```
sh inference.sh
```

你会看到如下信息：

```
I0303 20:41:54.096935  5255 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [memory_optimize_pass]
I0303 20:41:54.901712  5255 memory_optimize_pass.cc:216] Cluster name : relu_1.tmp_0  size: 307200
I0303 20:41:54.901755  5255 memory_optimize_pass.cc:216] Cluster name : relu_7.tmp_0  size: 7680000
I0303 20:41:54.901759  5255 memory_optimize_pass.cc:216] Cluster name : feat_input  size: 2457600
I0303 20:41:54.901762  5255 memory_optimize_pass.cc:216] Cluster name : matmul_v2_0.tmp_0  size: 245760000
I0303 20:41:54.901779  5255 memory_optimize_pass.cc:216] Cluster name : relu_5.tmp_0  size: 30720000
--- Running analysis [ir_graph_to_program_pass]
I0303 20:41:54.925014  5255 analysis_predictor.cc:714] ======= optimize end =======
I0303 20:41:54.925876  5255 naive_executor.cc:98] ---  skip [feed], feed -> feat_input
I0303 20:41:54.928025  5255 naive_executor.cc:98] ---  skip [sigmoid_5.tmp_0], fetch -> fetch
I0303 20:41:54.928069  5255 naive_executor.cc:98] ---  skip [squeeze_4.tmp_0], fetch -> fetch
I0303 20:41:54.928074  5255 naive_executor.cc:98] ---  skip [squeeze_7.tmp_0], fetch -> fetch
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [conv_affine_channel_fuse_pass]
--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [embedding_eltwise_layernorm_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v2]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
I0303 20:41:56.007589  5255 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
--- Running IR pass [fc_fuse_pass]
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
--- Running IR pass [conv_elementwise_add_act_fuse_pass]
--- Running IR pass [conv_elementwise_add2_act_fuse_pass]
--- Running IR pass [conv_elementwise_add_fuse_pass]
--- Running IR pass [transpose_flatten_concat_fuse_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I0303 20:41:56.014565  5255 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [memory_optimize_pass]
I0303 20:41:56.758029  5255 memory_optimize_pass.cc:216] Cluster name : relu_1.tmp_0  size: 307200
I0303 20:41:56.758069  5255 memory_optimize_pass.cc:216] Cluster name : relu_7.tmp_0  size: 7680000
I0303 20:41:56.758072  5255 memory_optimize_pass.cc:216] Cluster name : feat_input  size: 2457600
I0303 20:41:56.758075  5255 memory_optimize_pass.cc:216] Cluster name : matmul_v2_0.tmp_0  size: 245760000
I0303 20:41:56.758088  5255 memory_optimize_pass.cc:216] Cluster name : relu_5.tmp_0  size: 30720000
--- Running analysis [ir_graph_to_program_pass]
I0303 20:41:56.781632  5255 analysis_predictor.cc:714] ======= optimize end =======
I0303 20:41:56.782409  5255 naive_executor.cc:98] ---  skip [feed], feed -> feat_input
I0303 20:41:56.784457  5255 naive_executor.cc:98] ---  skip [sigmoid_5.tmp_0], fetch -> fetch
I0303 20:41:56.784487  5255 naive_executor.cc:98] ---  skip [squeeze_4.tmp_0], fetch -> fetch
I0303 20:41:56.784492  5255 naive_executor.cc:98] ---  skip [squeeze_7.tmp_0], fetch -> fetch
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [conv_affine_channel_fuse_pass]
--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [embedding_eltwise_layernorm_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v2]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
I0303 20:41:58.459947  5255 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
--- Running IR pass [fc_fuse_pass]
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
--- Running IR pass [conv_elementwise_add_act_fuse_pass]
--- Running IR pass [conv_elementwise_add2_act_fuse_pass]
--- Running IR pass [conv_elementwise_add_fuse_pass]
--- Running IR pass [transpose_flatten_concat_fuse_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I0303 20:41:58.466913  5255 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [memory_optimize_pass]
I0303 20:41:59.200733  5255 memory_optimize_pass.cc:216] Cluster name : relu_1.tmp_0  size: 307200
I0303 20:41:59.200773  5255 memory_optimize_pass.cc:216] Cluster name : relu_7.tmp_0  size: 7680000
I0303 20:41:59.200775  5255 memory_optimize_pass.cc:216] Cluster name : feat_input  size: 2457600
I0303 20:41:59.200778  5255 memory_optimize_pass.cc:216] Cluster name : matmul_v2_0.tmp_0  size: 245760000
I0303 20:41:59.200783  5255 memory_optimize_pass.cc:216] Cluster name : relu_5.tmp_0  size: 30720000
--- Running analysis [ir_graph_to_program_pass]
I0303 20:41:59.224738  5255 analysis_predictor.cc:714] ======= optimize end =======
I0303 20:41:59.225565  5255 naive_executor.cc:98] ---  skip [feed], feed -> feat_input
I0303 20:41:59.227782  5255 naive_executor.cc:98] ---  skip [sigmoid_5.tmp_0], fetch -> fetch
I0303 20:41:59.227814  5255 naive_executor.cc:98] ---  skip [squeeze_4.tmp_0], fetch -> fetch
I0303 20:41:59.227818  5255 naive_executor.cc:98] ---  skip [squeeze_7.tmp_0], fetch -> fetch
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [conv_affine_channel_fuse_pass]
--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [embedding_eltwise_layernorm_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v2]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
I0303 20:42:00.276561  5255 fuse_pass_base.cc:57] ---  detected 3 subgraphs
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
--- Running IR pass [fc_fuse_pass]
I0303 20:42:00.277803  5255 fuse_pass_base.cc:57] ---  detected 2 subgraphs
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
--- Running IR pass [conv_elementwise_add_act_fuse_pass]
--- Running IR pass [conv_elementwise_add2_act_fuse_pass]
--- Running IR pass [conv_elementwise_add_fuse_pass]
--- Running IR pass [transpose_flatten_concat_fuse_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I0303 20:42:00.284799  5255 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [memory_optimize_pass]
I0303 20:42:01.006960  5255 memory_optimize_pass.cc:216] Cluster name : relu_7.tmp_0  size: 7680000
I0303 20:42:01.007001  5255 memory_optimize_pass.cc:216] Cluster name : tmp_1  size: 307200
I0303 20:42:01.007011  5255 memory_optimize_pass.cc:216] Cluster name : feat_input  size: 2457600
I0303 20:42:01.007015  5255 memory_optimize_pass.cc:216] Cluster name : reshape2_4.tmp_0  size: 245760000
I0303 20:42:01.007035  5255 memory_optimize_pass.cc:216] Cluster name : matmul_v2_0.tmp_0  size: 245760000
--- Running analysis [ir_graph_to_program_pass]
I0303 20:42:01.037729  5255 analysis_predictor.cc:714] ======= optimize end =======
I0303 20:42:01.040416  5255 naive_executor.cc:98] ---  skip [feed], feed -> feat_input
I0303 20:42:01.041653  5255 naive_executor.cc:98] ---  skip [sigmoid_7.tmp_0], fetch -> fetch
I0303 20:42:01.041680  5255 naive_executor.cc:98] ---  skip [squeeze_5.tmp_0], fetch -> fetch
I0303 20:42:01.041684  5255 naive_executor.cc:98] ---  skip [squeeze_8.tmp_0], fetch -> fetch
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [conv_affine_channel_fuse_pass]
--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [embedding_eltwise_layernorm_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v2]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
I0303 20:42:02.069478  5255 fuse_pass_base.cc:57] ---  detected 3 subgraphs
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
--- Running IR pass [fc_fuse_pass]
I0303 20:42:02.070616  5255 fuse_pass_base.cc:57] ---  detected 2 subgraphs
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
--- Running IR pass [conv_elementwise_add_act_fuse_pass]
--- Running IR pass [conv_elementwise_add2_act_fuse_pass]
--- Running IR pass [conv_elementwise_add_fuse_pass]
--- Running IR pass [transpose_flatten_concat_fuse_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
I0303 20:42:02.077549  5255 ir_params_sync_among_devices_pass.cc:45] Sync params from CPU to GPU
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [memory_optimize_pass]
I0303 20:42:02.804873  5255 memory_optimize_pass.cc:216] Cluster name : relu_7.tmp_0  size: 7680000
I0303 20:42:02.804919  5255 memory_optimize_pass.cc:216] Cluster name : tmp_1  size: 307200
I0303 20:42:02.804922  5255 memory_optimize_pass.cc:216] Cluster name : feat_input  size: 2457600
I0303 20:42:02.804936  5255 memory_optimize_pass.cc:216] Cluster name : reshape2_4.tmp_0  size: 245760000
I0303 20:42:02.804947  5255 memory_optimize_pass.cc:216] Cluster name : matmul_v2_0.tmp_0  size: 245760000
--- Running analysis [ir_graph_to_program_pass]
I0303 20:42:02.835162  5255 analysis_predictor.cc:714] ======= optimize end =======
I0303 20:42:02.837682  5255 naive_executor.cc:98] ---  skip [feed], feed -> feat_input
I0303 20:42:02.838863  5255 naive_executor.cc:98] ---  skip [sigmoid_7.tmp_0], fetch -> fetch
I0303 20:42:02.838889  5255 naive_executor.cc:98] ---  skip [squeeze_5.tmp_0], fetch -> fetch
I0303 20:42:02.838893  5255 naive_executor.cc:98] ---  skip [squeeze_8.tmp_0], fetch -> fetch
W0303 20:42:02.864467  5255 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0303 20:42:02.869130  5255 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Current video file: /home/aistudio/data/Features_competition_test_B/npy/708697e09925421e80f28b0ddb6e69fd_19.npy :
{'score': 0.0, 'segment': [4.120000000000001, 4.2]}
{'score': 0.0, 'segment': [3.8800000000000003, 4.2]}
{'score': 0.0, 'segment': [3.84, 4.2]}
{'score': 0.0, 'segment': [0.7200000000000001, 2.16]}
{'score': 0.0, 'segment': [10.56, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/467ec2a9514a4c75b733ca1afd979ec9_23.npy :
{'score': 2.404912601193995e-19, 'segment': [11.56, 12]}
{'score': 3.57727017058996e-22, 'segment': [11.68, 12]}
{'score': 1.736600622769849e-36, 'segment': [11.96, 12]}
{'score': 2.2314276745908387e-41, 'segment': [10.68, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/9a366e6b8978477e8a3cd90c073d1f01_4.npy :
{'score': 0.0, 'segment': [5.16, 5.4]}
{'score': 0.0, 'segment': [2.4000000000000004, 2.8400000000000003]}
{'score': 0.0, 'segment': [4.6000000000000005, 5.4]}
{'score': 0.0, 'segment': [1.76, 2.8400000000000003]}
{'score': 0.0, 'segment': [4.16, 5.4]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/8fd4f6414e1e43b4bdce436a998e093a_6.npy :
{'score': 3.4987721506786695e-34, 'segment': [10.360000000000001, 12]}
{'score': 2.585921056409382e-35, 'segment': [10.32, 12]}
{'score': 8.116142635339218e-37, 'segment': [10.280000000000001, 12]}
{'score': 8.091442758364223e-38, 'segment': [10.4, 12]}
{'score': 2.978245581413949e-42, 'segment': [10.52, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/c65345af9a3d47a880dcaa12e8c5655d_4.npy :
{'score': 5.002517674043941e-35, 'segment': [0, 0.36000000000000004]}
{'score': 6.22951819501931e-36, 'segment': [0, 0.32]}
{'score': 4.271409952985621e-40, 'segment': [0, 0.9200000000000002]}
{'score': 1.66684452331437e-41, 'segment': [11.64, 12]}
{'score': 5.366973118364049e-43, 'segment': [10.72, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/b5543a88eded46b9bb8e5ad9346696cc_5.npy :
{'score': 0.0, 'segment': [2.12, 2.56]}
{'score': 0.0, 'segment': [9.200000000000001, 9.88]}
{'score': 0.0, 'segment': [1.6, 2.56]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/d41f7085f6d74cbca86d73b33a4e3f6d_28.npy :
{'score': 0.15520255267620087, 'segment': [0.04, 0.68]}
{'score': 0.038364309817552567, 'segment': [0.12, 1.04]}
{'score': 0.022063065000901303, 'segment': [0.12, 0.6000000000000001]}
{'score': 0.004574342916003852, 'segment': [0, 0.88]}
{'score': 0.004466861705727982, 'segment': [0, 0.44]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/1841993a73504e9e8624b728133b81ff_17.npy :
{'score': 3.443353584572417e-27, 'segment': [11.2, 12]}
{'score': 1.6876401242778102e-30, 'segment': [11.84, 12]}
{'score': 5.438225061420675e-33, 'segment': [11.520000000000001, 12]}
{'score': 0.0, 'segment': [7.880000000000001, 8.08]}
{'score': 0.0, 'segment': [5.24, 6.32]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/07bb92c132974cb7adb4b0ad19a87833_7.npy :
{'score': 2.6739565125553717e-27, 'segment': [11.68, 12]}
{'score': 2.255221050280595e-28, 'segment': [11.64, 12]}
{'score': 1.6916560966281527e-29, 'segment': [11.72, 12]}
{'score': 1.1004010633527755e-29, 'segment': [11.32, 12]}
{'score': 5.27379604816802e-31, 'segment': [11.6, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/d25ab25a350842a1b8e2e1695c244299_22.npy :
{'score': 0.0, 'segment': [6.08, 6.440000000000001]}
{'score': 0.0, 'segment': [5.8, 6.440000000000001]}
{'score': 0.0, 'segment': [4.0, 5.040000000000001]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/3420c6f5635541ffbd0857a1d171f9a7_16.npy :
{'score': 0.19647204875946045, 'segment': [0.2, 0.88]}
{'score': 0.059052977710962296, 'segment': [0.16, 0.6000000000000001]}
{'score': 0.044617743070176595, 'segment': [0.24, 1.12]}
{'score': 0.027710991823068798, 'segment': [0.28, 0.76]}
{'score': 0.010486319548609003, 'segment': [0.08, 0.64]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/a946b48ac6a746f58cc768cd6255bca0_7.npy :
{'score': 0.12320282310247421, 'segment': [10.8, 11.4]}
{'score': 0.01620224795405116, 'segment': [10.760000000000002, 11.440000000000001]}
{'score': 0.0045006051659584045, 'segment': [9.96, 11.4]}
{'score': 0.0033118586040658178, 'segment': [10.64, 11.360000000000001]}
{'score': 0.0008773504747711682, 'segment': [10.840000000000002, 11.48]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/8818f8ad29e646c68f182eb4f70e717f_19.npy :
{'score': 0.20745404064655304, 'segment': [5.880000000000001, 6.440000000000001]}
{'score': 0.16207842528820038, 'segment': [6.960000000000001, 7.48]}
{'score': 0.04189765591272418, 'segment': [6.0, 6.48]}
{'score': 0.03708048189170011, 'segment': [7.0, 7.600000000000001]}
{'score': 0.02973596565425396, 'segment': [6.840000000000001, 7.32]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/f2289c4496f648f981c18aa6b270c690_22.npy :
{'score': 0.19432373344898224, 'segment': [9.040000000000001, 9.64]}
{'score': 0.030185025614771793, 'segment': [9.08, 9.72]}
{'score': 0.024143101647496223, 'segment': [8.84, 9.48]}
{'score': 0.00971383135765791, 'segment': [10.0, 10.52]}
{'score': 0.006661587822077956, 'segment': [9.120000000000001, 9.56]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/35fa56fbd11749bd8db4f82558b20b8a_21.npy :
{'score': 0.3007500469684601, 'segment': [2.7600000000000002, 3.4400000000000004]}
{'score': 0.2240048497915268, 'segment': [3.84, 4.36]}
{'score': 0.19496119022369385, 'segment': [1.6, 2.3600000000000003]}
{'score': 0.06941152722471472, 'segment': [3.7600000000000002, 4.5600000000000005]}
{'score': 0.05833832612677236, 'segment': [2.8, 3.3200000000000003]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/8147225acc6a4a72adfca1906b9fbadc_1.npy :
{'score': 0.1685645431280136, 'segment': [0.68, 1.2000000000000002]}
{'score': 0.15728989243507385, 'segment': [1.56, 2.08]}
{'score': 0.039813652634620667, 'segment': [1.4400000000000002, 2.4000000000000004]}
{'score': 0.028966497088103345, 'segment': [0.7200000000000001, 1.12]}
{'score': 0.025887785479426384, 'segment': [1.48, 1.8800000000000001]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/3c9fd822fdaa4fc0ab6d6c25a831dc0b_6.npy :
{'score': 0.0, 'segment': [11.32, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/bba9bfc0236c4cf0a7c45cfa5a0e5e69_23.npy :
Current video file: /home/aistudio/data/Features_competition_test_B/npy/f2289c4496f648f981c18aa6b270c690_7.npy :
{'score': 5.201084903689518e-22, 'segment': [3.84, 4.5200000000000005]}
{'score': 2.774624641357445e-23, 'segment': [3.84, 4.5600000000000005]}
{'score': 1.654357426190014e-24, 'segment': [3.84, 4.48]}
{'score': 1.5344241976630381e-25, 'segment': [3.84, 4.6000000000000005]}
{'score': 1.5260475692901586e-26, 'segment': [2.8, 4.36]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/1609d62a3ec2450bb17468dd17ff9191_2.npy :
{'score': 0.26886099576950073, 'segment': [4.16, 4.88]}
{'score': 0.22856837511062622, 'segment': [6.279999999999999, 6.720000000000001]}
{'score': 0.18188756704330444, 'segment': [5.4, 5.8]}
{'score': 0.17457406222820282, 'segment': [7.080000000000001, 7.600000000000001]}
{'score': 0.05084540694952011, 'segment': [7.04, 8.0]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/5192c733730446bfba0c0f2fe722e847_28.npy :
{'score': 1.1210387714598537e-44, 'segment': [11.96, 12]}
{'score': 0.0, 'segment': [2.92, 3.2]}
{'score': 0.0, 'segment': [3.7199999999999998, 4.640000000000001]}
{'score': 0.0, 'segment': [8.16, 9.440000000000001]}
{'score': 0.0, 'segment': [0, 1.4]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/e7d593344d174e4da41f5af61bf56fe5_2.npy :
{'score': 1.064986832886861e-43, 'segment': [11.96, 12]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/210dda53a231423aa0202e5c4b8182f7_15.npy :
{'score': 0.0, 'segment': [0, 0.08]}
{'score': 0.0, 'segment': [2.3200000000000003, 3.64]}
{'score': 0.0, 'segment': [1.6, 3.4400000000000004]}
{'score': 0.0, 'segment': [2.08, 3.8800000000000003]}
{'score': 0.0, 'segment': [3.4800000000000004, 5.24]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/210dda53a231423aa0202e5c4b8182f7_21.npy :
{'score': 0.22425995767116547, 'segment': [9.8, 10.360000000000001]}
{'score': 0.13130789995193481, 'segment': [10.92, 11.32]}
{'score': 0.11109349876642227, 'segment': [9.56, 10.24]}
{'score': 0.10323692113161087, 'segment': [11.64, 12]}
{'score': 0.07797395437955856, 'segment': [10.64, 11.280000000000001]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/6849f9d2f2fa4a27a6e0ba47037479d8_3.npy :
{'score': 0.08702588081359863, 'segment': [9.84, 10.52]}
{'score': 0.010667069782440447, 'segment': [9.76, 10.48]}
{'score': 0.0038123278672308556, 'segment': [9.96, 10.68]}
{'score': 0.002019177841111469, 'segment': [9.88, 10.4]}
{'score': 0.0013432508567348123, 'segment': [9.200000000000001, 10.48]}
Current video file: /home/aistudio/data/Features_competition_test_B/npy/85a24864079f4456894537f3186fdbbf_28.npy :
{'score': 1.4575886951953033e-14, 'segment': [11.68, 12]}
{'score': 6.117814042634461e-17, 'segment': [11.72, 12]}
{'score': 3.0457943510079003e-19, 'segment': [11.760000000000002, 12]}
{'score': 1.6371282700710647e-21, 'segment': [11.8, 12]}
{'score': 7.360249984729093e-24, 'segment': [11.84, 12]}
```


## 11、验证集后处理

```
sh val_postprocess.sh
```

测试脚本运行完成后，可以在当前目录中得到submission1.zip文件，将该文件提交至评测官网，即可以查看在B榜得分。

在原始路径下存放了我们B榜的提交文件submission.zip。

最后强烈推荐PaddleVideo，欢迎大家Star⭐收藏一下～
![](https://ai-studio-static-online.cdn.bcebos.com/a0d0420262e9484fb2eaf428f8fe03b0cd2d32f30f7e4192a2126ee1e5fc6fc3)
