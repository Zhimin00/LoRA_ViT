## RETFound - A foundation model for retinal imaging


1. Create environment with conda:

```
conda create -n retfound python=3.11.0 -y
conda activate retfound
```

2. Install dependencies

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/rmaphoh/RETFound_MAE/
cd RETFound_MAE
pip install -r requirements.txt
```


### 🌱Fine-tuning with RETFound weights

To fine tune RETFound on your own data, follow these steps:

1. Get access to the pre-trained models on HuggingFace (register an account and fill in the form) and go to step 2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureCFP</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureOCT</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureOCT">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_meh">access</a></td>
<td align="center">TBD</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_shanghai">access</a></td>
<td align="center">TBD</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_meh">access</a></td>
<td align="center">TBD</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center">TBD</a></td>
</tr>
</tbody></table>

2. Login in your HuggingFace account, where HuggingFace token can be [created and copied](https://huggingface.co/settings/tokens).
```
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```

**Optional**: if your machine and server cannot access HuggingFace due to internet wall, run the command below (Do not run it if you can access):
```
export HF_ENDPOINT=https://hf-mirror.com
```

3. Organise your data into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 

4. Start fine-tuning (use IDRiD as example). A fine-tuned checkpoint will be saved during training. Evaluation will be automatically run after training.

The model and finetune can be selected:

| model           | finetune                 |
|-----------------|--------------------------|
| RETFound_mae    | RETFound_mae_natureCFP   |
| RETFound_mae    | RETFound_mae_natureOCT   |
| RETFound_mae    | RETFound_mae_meh         |
| RETFound_mae    | RETFound_mae_shanghai    |
| RETFound_dinov2 | RETFound_dinov2_meh      |
| RETFound_dinov2 | RETFound_dinov2_shanghai |

```
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 16 \
    --world_size 1 \
    --epochs 100 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD \
    --input_size 224 \
    --task RETFound_mae_meh-IDRiD \
    --finetune RETFound_mae_meh
```



