# MedDr: Diagnosis-Guided Bootstrapping for Large-Scale Medical Vision-Language Learning

*A generalist foundation model for healthcare capable of handling diverse medical data modalities.*

<div align=left>
<img src=examples/logo.jpg width=15% />
</div>


 [[Project Page](https://smart-meddr.github.io/)] [[🤗Model](https://huggingface.co/Sunanhe/MedDr_0401)] 

**MedDr: Diagnosis-Guided Bootstrapping for Large-Scale Medical Vision-Language Learning** [[Paper](https://arxiv.org/abs/2404.15127)] <br>
[Sunan He*](https://jerrrynie.github.io/), [Yuxiang Nie*](https://jerrrynie.github.io/), [Zhixuan Chen](https://zhi-xuan-chen.github.io/homepage/), [Zhiyuan Cai](https://github.com/Davidczy), Hongmei Wang, [Shu Yang](https://github.com/isyangshu), [Hao Chen**](https://cse.hkust.edu.hk/~jhc/) (*Equal Contribution, **Corresponding author)


## Release
- [04/23] 🔥 We released **MedDr: Diagnosis-Guided Bootstrapping for Large-Scale Medical Vision-Language Learning**. We developed MedDr, a generalist foundation model for healthcare capable of handling diverse medical data modalities, including radiology, pathology, dermatology, retinography, and endoscopy. Check out the [paper](https://arxiv.org/abs/2404.15127).


## Schedule

+ [x] Release the demo code.
+ [ ] Release the evaluation code.
+ [ ] Release the training code.
+ [ ] Release the data generation code.


## Environment

We build our model based on [InternVL](https://github.com/OpenGVLab/InternVL). Please refer to the [INSTALLATION.md](https://github.com/OpenGVLab/InternVL/blob/main/INSTALLATION.md) to prepare the environment.


## Demo
Download the [checkpoint](https://huggingface.co/Sunanhe/MedDr_0401) and change the `model_path` in `demo.py`.

```Shell
python3 demo.py
```

## Acknowledgement

- [InternVL](https://github.com/OpenGVLab/InternVL): 
Thanks for their efforts in the open-source community. InternVL is a highly valuable work that contributes significantly to the VLM domain.

