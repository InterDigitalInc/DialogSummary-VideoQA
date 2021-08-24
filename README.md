## On the hidden treasure of dialog in video question answering

[Deniz Engin](https://engindeniz.github.io/), [François Schnitzler](https://sites.google.com/site/francoisschnitzler/), [Ngoc Q. K. Duong](https://www.interdigital.com/talent/?id=88) and [Yannis Avrithis](https://avrithis.net/), On the hidden treasure of dialog in video question answering, ICCV 2021. 

[Project page](https://engindeniz.github.io/dialogsummary-videoqa) | [arXiv](https://arxiv.org/abs/2103.14517)

---
### Model Overview
![Model](images/model.png?raw=true)


Our VideoQA system converts dialog and video inputs to episode dialog summaries and video descriptions, respectively. Converted inputs and dialog are processed independently in streams, along with the question and each answer,
producing a score per answer. Finally, stream embeddings are fused separately per answer and a prediction is made.


### Environment Setup
````
conda create --name  dialog-videoqa python=3.6
conda activate dialog-videoqa
conda install -c anaconda numpy pandas scikit-learn
conda install -c conda-forge tqdm
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
pip install pytorch-transformers
````

### Data preparation
* Download [KnowIT VQA](https://knowit-vqa.github.io/) dataset and extract in [data folder](data).
* Extracted scene and episode dialog summaries are provided as separate files in [data folder](data).
* Plot summaries used in [ROLL-VideoQA](https://arxiv.org/pdf/2007.08751.pdf), they can be download from [here](https://github.com/noagarcia/ROLL-VideoQA/blob/master/Data/knowledge_base/tbbt_summaries.csv).
* Scene descriptions are obtained by following [ROLL-VideoQA](https://github.com/noagarcia/ROLL-VideoQA), generated descriptions are provided in [data folder](data).

### Training Models 

This section explains single-stream QA and multi-stream QA trainings.

##### Single-Stream QA

Our main streams are video, scene dialog summary, episode dialog summary. Dialog and plot streams are used for comparison. All stream trainings as follows: 

Training video stream:
```
python stream_main_train.py --train_name video --max_seq_length 512
```
Training scene dialog summary stream:
```
python stream_main_train.py --train_name scene_dialog_summary --max_seq_length 512
```
Training episode dialog summary stream:
```
python stream_main_train.py --train_name episode_dialog_summary --max_seq_length 300 --seq_stride 200 --mini_batch_size 2 --eval_batch_size 16 
```
Training dialog stream:
```
python stream_main_train.py --train_name dialog --max_seq_length 512

```
Training plot stream:
```
python stream_main_train.py --train_name plot --max_seq_length 200 --seq_stride 100 --mini_batch_size 2
```

All single stream models trained on 2 Tesla V100 GPUs (32 GB) except plot trained on 1 Tesla V100.
Gradient accumulation is used to fit into memory when training parameters have a "mini_batch_size".

#### Multi-Stream QA

Our main proposed model uses video, scene dialog summary and episode dialog summary streams for multi-stream attention method.
``` 
python fusion_main_train.py --fuse_stream_list video scene_dialog_summary episode_dialog_summary --fusion_method multi-stream-attention
```

### License

Please check the [license file](license.txt) for more information. 

### Acknowledgments

The code is written based on <a href="https://github.com/noagarcia/ROLL-VideoQA" target="_blank">ROLL-VideoQA</a>. 

### Citation

If this code is helpful for you, please cite the following: 

````
@inproceedings{engin2021hidden,
  title={On the hidden treasure of dialog in video question answering},
  author={Engin, Deniz and Schnitzler, Fran{\c{c}}ois and Duong, Ngoc QK and Avrithis, Yannis},
  journal={ICCV},
  year={2021}
}

````
