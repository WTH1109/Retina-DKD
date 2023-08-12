# Multimodal transformer system for diabetic nephropathy diagnosis based on fundus image

- This is the official repository of the paper "Multimodal transformer system for diabetic nephropathy diagnosis based on fundus image"

<img src="img\model.png" alt="TransMYF" style="zoom:50%;" />

## 1. Environment

See `requirements.txt`



## 2. Dataset

1. The training data and testing data is from the XXX, we will provide `data` and `data_test` , and pre-trained models.

2. Put `data` and `data_test`  under the root directory 

3. You can choose to build your own training set, named `cls` + `peugeot` . You can change `-d` to specify the training set while training and testing.  But you need to ensure that its file structure is as follows

   <img src="img\tree.jpg" style="zoom: 30%;" />



## 3. Training

The details of the hyper-parameters are all listed in the `train_fusion.py`. Use the below command to train our fusion model on the Retina-DKD database.

```sh
python train_fusion.py -b TransMUF -g 0 -bc 8 -e 150 -d _dkd
```

If you want a more detailed training method, modify the configuration file in `config/train_config.py`.

It is worth noting that  `-d` represents the `peugeot` of the database. 

eg.  If you name the database as `cls_dkd`，then `-d`  will be `_dkd` 



If you want to train a model that uses only fundus images, use the below command.

```sh
python train_cls.py -b resnet-wam -g 0 -bc 8 -e 150 -d _dkd
```



## 4. Test

Modify the config file  `config/test_config.yaml` to select a different test configuration

After that you can run `python test_run.py` to run the test code

Beside, we will provide a pre-trained TransMUF model and a segmented network model, and place them under the `model/`,`code_seg/model/` .



### test config

In config, you need to specify the **save_name** of the model, the **specific model**, and the **epoch** of the test and **related parameters**.

You can run different test files by modifying the `select_num`

```python
python_name_set = [
    'test_isolate_all.py',  # 0 test uses only fundus images
    'test_fusion.py',       # 1 test fusion model 
    'test_fusion_pat.py',   # 2 test fusion model in patient-wise
    'grad-cam-fusion.py',   # 3 generate the visualize image
]
```

To test our results on different databases, change the value of `dataset_select`

Specific corresponding results of  `dataset_select`  are as follows:

```python
dataset_r = ['main', 		  # 0
	'Prospective',   # 1
	'Multi_center',  # 2
	'Non_standard']  # 3
```



**Below is a specific configuration file**

```yaml
select_num: 1
model_name_save_r: 'transMUF'

basic_model: 'TransMUF'
dataset_supple: _dkd

ep_true: 25
win_num: 3
lb_weight: 1

xlsx_name_r: 'runs_result.xls'
dataset_select: [0,1,2,3]

gpu: 0
```



## 5. Segment network

See Folder `code_seg`



## 6. Compared Methods and Ablation Study

If you are interested in our ablation study, the code for the comparison and ablation experiments is included in the `network.py` .

The specific meaning of the model name is written in the network comment



You just need to specify specific models during training and testing.

eg.  Run the following code to train TransMUF that do not contain clinical metrics

```sh
python train_fusion.py -b m1 -g 0 -bc 8 -e 150 -d _dkd
```



## 7. Network Interpretability

Run the grad-cam test program to get a visual image of the network (see Test)

The visualization will be placed under the `Vis_result` folder

<img src="img\vis.png" style="zoom: 80%;" />



## 8. Contact

If any question, please contact [[wengtaohan1109@gmail.com](wengtaohan1109@gmail.com)]