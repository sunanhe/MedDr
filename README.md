# GSCo: Towards generalizable AI in medicine via Generalist–Specialist Collaboration

*A cooperative framework for generalist-specialist collaboration in medical AI.*

<!-- markdownlint-disable MD033 -->
<img src="examples/logo.jpg" alt="GSCo logo" width="150">
<!-- markdownlint-enable MD033 -->

[[Paper](https://www.nature.com/articles/s41551-026-01653-3)] [[MedDr](https://huggingface.co/Sunanhe/MedDr_0401)] [[Specialist Models](https://huggingface.co/Sunanhe/GSCo_Specialist)]

## Project Overview

GSCo is an innovative medical AI framework that achieves better performance through Generalist-Specialist collaboration. This project consists of three main components:

1. **MedDr (Generalist Foundation Model)**: An open-source medical generalist foundation model.
2. **Specialist Models**: Task-specific expert models optimized for medical image classification and report generation.
3. **GSCo Framework**: A collaborative framework that combines the strengths of Generalist and Specialist models.

### Key Features

- **Medical Multimodal**: Supports medical image analysis and report generation.
- **Collaborative Inference**: Enhances performance through Generalist-Specialist cooperation.
- **Modular Design**: Supports different specialist models and datasets.

## Quick Start

### Environment Setup

We build our model based on [InternVL](https://github.com/OpenGVLab/InternVL). Please refer to the [INSTALLATION.md](https://github.com/OpenGVLab/InternVL/blob/main/INSTALLATION.md) to prepare the environment.

### Basic Usage Pipeline

1. **Demo Experience**:

   ```bash
   python3 demo.py
   ```

2. **Model Training**:

   ```bash
   sh train.sh
   ```

3. **Model Evaluation**:

   ```bash
   # Evaluate MedDr foundation model
   sh inference_meddr.sh

   # Generate Specialist predictions
   sh inference_specialist.sh

   # Evaluate GSCo collaborative framework
   sh inference_gsco.sh
   ```

## Demo

Download the [checkpoint](https://huggingface.co/Sunanhe/MedDr_0401) and change the `model_path` in `demo.py`. The demo will be finished in 5 seconds (on a H800 GPU).

```Shell
python3 demo.py
```

## Training

### Data Preparation

We follow the format of [InternVL](https://github.com/OpenGVLab/InternVL) to prepare the data.
For example, the data format is as follows:

```json
{
    "id": 0,
    "study_id": 50414267,
    "subject_id": 10000032,
    "split": "train",
    "image": "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nYou are a helpful medical assistant. Your task is report generation. You are given a chest x-ray image and you are required to generate a summary report about the image."
        },
        {
            "from": "gpt",
            "value": "There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted."
        }
    ]
}
```

We also provide an example file `data/meddr.json` for your reference.
Please find more details about the datasets involved in the [DATASET.md](DATASET.md).

### Generalist Foundation Model

You need at least two 80GB GPUs (e.g., NVIDIA H800 GPU) to train the Generalist Foundation Model.

```Shell
sh train.sh
```

We provide the checkpoint of [MedDr](https://hkustconnect-my.sharepoint.com/:f:/g/personal/shebd_connect_ust_hk/EjJ6KFsPb0VEgu7tZpSUFb0BiTwpZWG0sW2qg4faZzp_tA?e=fKvfm0) here.

### Specialist Model

#### Medical Image Classification

We follow the training process in [Rethinking Model Prototyping MedMNIST+](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus) and extend it to more datasets. Training can be finished on a single 24GB GPU (e.g., NVIDIA RTX 4090 GPU).

Please find the checkpoints of the specialist models on [GSCo_Specialist](https://huggingface.co/Sunanhe/GSCo_Specialist).

#### Medical Report Generation

We employ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) as our specialist model for medical report generation. We use the [Official Checkpoint](https://drive.google.com/drive/folders/1ywEITWfYIAAYy0VY1IZ24Ec_GoNmkqIY) of R2GenGPT.

## Evaluation

### MedDr (Generalist Foundation Model)

In this section, we evaluate the performance of MedDr, our generalist foundation model that can handle various medical imaging tasks.

We provide the test data of [IU-XRay dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/shebd_connect_ust_hk/EfjDe98kRodPksP6YN4sT_0B13HmO9aWAgH30u78ic4auA?e=TuHibW) for reproduction.
Please download the data and change the path of the dataset accordingly.
The metafile used in this demo is `data/iu_mrg_meta.jsonl`.
You need at least one 80GB GPU (e.g., NVIDIA H800 GPU) to evaluate the Generalist Foundation Model.

```Shell
sh inference_meddr.sh
```

### Specialist Prediction Generation

Before evaluating GSCo, please generate predictions from Specialist models with the following script.

```Shell
sh inference_specialist.sh
```

You can modify the script parameters to adapt to different datasets and model architectures:

- `DATASET`: Specify the dataset to use
- `ARCH`: Specify the model architecture
- Additional parameters can be configured in `src/config/config.yaml`

You can find the data of PCam200 curated for the specialist model on [PCam200](https://pan.baidu.com/s/1jecaUDhdGDhpWnuSQVUaFg?pwd=GSCo) (password: GSCo).
The metafile and the result of PCam200 dataset can be find on [PCam200 Google Drive](https://drive.google.com/drive/folders/1HMpdpdaOOr8RrAZnw__DzH_vaZCcFg3i?usp=sharing)

### GSCo (Generalist-Specialist Collaboration)

After generating Specialist predictions, you can evaluate the performance of GSCo. GSCo achieves better medical AI performance through a Generalist-Specialist collaborative framework.
The metafile used in this demo is `data/pcam200_meta.json`.
You can find the data of PCam200 on [PCam200](https://pan.baidu.com/s/1fGNCTcTS4J9ftay4HXk1Mg?pwd=GSCo) (password: GSCo).
You need at least one 80GB GPU (e.g., NVIDIA H800 GPU) to evaluate the GSCo Framework.

```Shell
sh inference_gsco.sh
```

## Acknowledgement

- [InternVL](https://github.com/OpenGVLab/InternVL):
Thanks for their efforts in the open-source community. InternVL is a highly valuable work that contributes significantly to the VLM domain.
- [Rethinking Model Prototyping MedMNIST+](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus):
Thanks for their implementation of training and evaluation on MedMNIST+.
- [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT):
Thanks for their great work in the medical report generation task.

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{he2026generalizable,
  title={Towards generalizable AI in medicine via Generalist-Specialist Collaboration},
  author={Sunan He and Yuxiang Nie and Hongmei Wang and Shu Yang and Yihui Wang and Zhiyuan Cai and Zhixuan Chen and Yingxue Xu and Luyang Luo and Huiling Xiang and Xi Lin and Mingxiang Wu and Yifan Peng and George Shih and Ziyang Xu and Xian Wu and Qiong Wang and Ronald Cheong Kin Chan and Xiaohui Duan and Varut Vardhanabhuti and Winnie Chiu Wing Chu and Yefeng Zheng and Pranav Rajpurkar and Kang Zhang and Hao Chen},
  journal={Nature Biomedical Engineering},
  year={2026},
  doi={10.1038/s41551-026-01653-3},
  url={https://www.nature.com/articles/s41551-026-01653-3}
}
```
