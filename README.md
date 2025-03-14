<div align="center">

<img src="figs/easydetect.jpg" width="18%" height="18%">

**An Easy-to-Use Multimodal Hallucination Detection Framework for MLLMs**


---
$\color{red}{It\'s\  unfortunate\  that\  due\  to\  limited\  computational\  resources,\  we\  have\  suspended\  the\  online\  demo.}$
$\color{red}{If\  you\'d\  like\  to\  try\  the\  demo,\  please\  contact\  sunnywcx \@zju.edu.cn\  or\  zhangningyu \@zju.edu.cn}$


<p align="center">
  <a href="#citation">🌻Acknowledgement</a> •
  <a href="https://huggingface.co/datasets/openkg/MHaluBench">🤗Benchmark</a> •
  <a href="http://easydetect.zjukg.cn/">🍎Demo</a> •
  <a href="#overview">🌟Overview</a> •
  <a href="#modelzoo">🐧ModelZoo</a> •
  <a href="#installation">🔧Installation</a> •
  <a href="#quickstart">⏩Quickstart</a> •
  <a href="#citation">🚩Citation</a> 
  <!-- <a href="#contributors">🎉Contributors</a> -->
</p>

![](https://img.shields.io/badge/version-v0.1.1-blue)
![](https://img.shields.io/github/last-commit/zjunlp/EasyDetect?color=green) 
![](https://img.shields.io/badge/PRs-Welcome-red) 
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) -->

</div>

## Table of Contents

- <a href="#acknowledgement">🌻Acknowledgement</a>
- <a href="#overview">🌟Overview</a>
  - <a href="#unified-multimodal-hallucination">Unified Multimodal Hallucination </a>
  - <a href="#dataset-mhallubench-statistic">Dataset: MHalluBench Statistic</a>
  - <a href="#framework-uniHD-illustration">Framework: UniHD Illustration</a>
- <a href="#modelzoo">🐧ModelZoo</a>
- <a href="#installation">🔧Installation</a>
- <a href="#quickstart">⏩Quickstart</a>
- <a href="#citation">🚩Citation</a>
---
## 🔔News
## News
- **2024-05-17 The paper [Unified Hallucination Detection for Multimodal Large Language Models](https://arxiv.org/abs/2402.03190) is accepted by ACL 2024 main conference.**
- **2024-04-21 We replace all the base models in the demo with our own trained models, significantly reducing the inference time.**
- **2024-04-21 We release our open-source hallucination detection model HalDet-LLAVA, which can be downloaded in huggingface, modelscope and wisemodel.**
- **2024-02-10 We release the EasyDetect [demo](http://easydetect.zjukg.cn/)**.
- **2024-02-05  We release the paper:"[Unified Hallucination Detection for Multimodal Large Language Models](https://arxiv.org/abs/2402.03190)" with a new benchmark [MHaluBench](https://huggingface.co/datasets/openkg/MHaluBench)! We are looking forward to any comments or discussions on this topic :)**
- **2023-10-20 The EasyDetect project has been launched and is under development.**


## 🌻Acknowledgement

Part implementation of this project were assisted and inspired by the related hallucination toolkits including [FactTool](https://github.com/GAIR-NLP/factool), [Woodpecker](https://github.com/BradyFU/Woodpecker), and others. 
This repository also benefits from the public project from [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [MAERec ](https://github.com/Mountchicken/Union14M). 
We follow the same license for open-sourcing and thank them for their contributions to the community.



## 🌟Overview

EasyDetect is a systematic package which is proposed as an easy-to-use hallucination detection framework for Multimodal Large Language Models(MLLMs) like GPT-4V, Gemini, LlaVA in your research experiments. 

### Unified Multimodal Hallucination

#### Unified View of Detection

A prerequisite for unified detection is the coherent categorization of the principal categories of hallucinations within MLLMs. Our paper superficially examines the following Hallucination Taxonomy from a unified perspective:

<p align="center">
<img src="figs/view.png"  width="60%" height="60%">
<img src="figs/intro.png" width="60%" height="60%">
</p>

**Figure 1:** Unified multimodal hallucination detection aims to identify and detect modality-conflicting hallucinations at
various levels such as object, attribute, and scene-text, as well as fact-conflicting hallucinations in both image-to-text and text-to-image generation.

**Modality-Conflicting Hallucination.**  MLLMs sometimes generate outputs that conflict with inputs from other modalities, leading to issues such as incorrect objects, attributes, or scene text. An example in above Figure (a) includes an MLLM inaccurately describing an athlete's uniform , showcasing an attribute-level conflict due to MLLMs' limited ability to achieve fine-grained text-image alignment.

**Fact-Conflicting Hallucination.** Outputs from MLLMs may contradict established factual knowledge. Image-to-text models can generate narratives that stray from the actual content by incorporating irrelevant facts, while text-to-image models may produce visuals that fail to reflect the factual knowledge contained in text prompts. These discrepancies underline the struggle of MLLMs to maintain factual consistency, representing a significant challenge in the domain.

#### Fine-grained Detection Task Definition

Unified detection of multimodal hallucination necessitates the check of each image-text pair `a={v, x}`, wherein `v` denotes either the visual input provided to an MLLM, or the visual output synthesized by it. Correspondingly, `x` signifies the MLLM's generated textual response based on `v` or the textual user query for synthesizing `v`. Within this task, each `x` may contain multiple claims, denoted as $\{c_i\}\_\{i = 1 \cdots n\}$. The objective for hallucination detectors is to assess each claim from `a` to determine whether it is "hallucinatory" or "non-hallucinatory", providing a rationale for their judgments based on the provided definition of hallucination. Text hallucination detection from LLMs denotes a sub-case in this setting, where `v` is null.

### Dataset: MHalluBench Statistic

To advance this research trajectory, we introduce the meta-evaluation benchmark MHaluBench, which encompasses the content from image-to-text and text-to-image generation, aiming to rigorously assess the advancements in multimodal halluci-
nation detectors. Further statistical details about MHaluBench are provided in below Figures.

<img src="figs/datasetinfo.jpg">

**Table 1:** *A comparison of benchmarks with respect to existing fact-checking or hallucination evaluation.* "Check." indicates verifying factual consistency, "Eval." denotes evaluating hallucinations generated by different LLMs, and its response is based on different LLMs under test, while "Det." embodies the evaluation of a detector’s capability in identifying hallucinations.

<p align="center">
  <img src="figs/饼图.png" width="40%" height="40%">
</p>

**Figure 2:** *Claim-Level data statistics of MHaluBench.* "IC" signifies Image Captioning and "T2I" indicates Text-to-Image synthesis, respectively.

<p align="center">
<img src="figs/条形图.png"   width="50%" height="50%">
</p>

**Figure 3:** *Distribution of hallucination categories within hallucination-labeled claims of MHaluBench.* 

### Framework: UniHD Illustration

Addressing the key challenges in hallucination detection, we introduce a unified framework in Figure 4 that systematically tackles multimodal hallucination identification for both image-to-text and text-to-image tasks. Our framework capitalizes on the domain-specific strengths of various tools to efficiently gather multi-modal evidence for confirming hallucinations. 

<img src="figs/framework.png">

**Figure 4:** *The specific illustration of UniHD for unified multimodal hallucination detection.* 

---

## 🐧ModelZoo
You can download two versions of HalDet-LLaVA, 7b and 13b, on three platforms: HuggingFace, ModelScope, and WiseModel.
| HuggingFace |  ModelScope |  WiseModel  |
| ----------- | ----------- | ----------- |
| [HalDet-llava-7b](https://huggingface.co/zjunlp/HalDet-llava-7b)     | [HalDet-llava-7b](https://www.modelscope.cn/models/ZJUNLP/HalDet-llava-7b)       | [HalDet-llava-7b](https://www.wisemodel.cn/models/zjunlp/HalDet-llava-7b)       |
| [HalDet-llava-13b](https://huggingface.co/zjunlp/HalDet-llava-13b)   | [HalDet-llava-13b](https://www.modelscope.cn/models/ZJUNLP/HalDet-llava-13b)        | [HalDet-llava-13b](https://www.wisemodel.cn/models/zjunlp/HalDet-llava-13b)      |

The claim level results on validation dataset
- Self-Check(GPT-4V) means use GPT-4V with 0 or 2 cases
- UniHD(GPT-4V/GPT-4o) means use GPT-4V/GPT-4o with 2-shot and tool information
- HalDet (LLAVA) means use LLAVA-v1.5 trained on our train datasets
<table>
    <tr>
        <td>task type</td>
        <td>model</td>
        <td>Acc</td>
        <td>Prec avg</td>
        <td>Recall avg</td>
        <td>Mac.F1</td>
    </tr>
    <tr>
        <td rowspan="6">image-to-text</td>
        <td>Self-Check 0shot (GPV-4V)</td>
        <td>75.09</td> 
        <td>74.94</td>
        <td>75.19</td>
        <td>74.97</td>
    </tr>
    <tr>
        <td>Self-Check 2shot (GPV-4V)</td>
        <td>79.25</td>
        <td>79.02</td>
        <td>79.16</td>
        <td>79.08</td>
    </tr>
    <tr>
        <td>HalDet (LLAVA-7b)</td>
        <td>75.02</td>
        <td>75.05</td>
        <td>74.18</td>
        <td>74.38</td>
    </tr>
    <tr>
        <td>HalDet (LLAVA-13b)</td>
        <td>78.16</td>
        <td>78.18</td>
        <td>77.48</td>
        <td>77.69</td>
    </tr>
    <tr>
        <td>UniHD(GPT-4V)</td>
        <td>81.91</td>
        <td>81.81</td>
        <td>81.52</td>
        <td>81.63</td>
    </tr>
    <tr>
        <td>UniHD(GPT-4o)</td>
        <td>86.08</td>
        <td>85.89</td>
        <td>86.07</td>
        <td>85.96</td>
    </tr>
    <tr>
        <td rowspan="6">text-to-image</td>
        <td>Self-Check 0shot (GPV-4V)</td>
        <td>76.20</td>
        <td>79.31</td>
        <td>75.99</td>
        <td>75.45</td>
    </tr>
    <tr>
        <td>Self-Check 2shot (GPV-4V)</td>
        <td>80.76</td>
        <td>81.16</td>
        <td>80.69</td>
        <td>80.67</td>
    </tr>
    <tr>
        <td>HalDet (LLAVA-7b)</td>
        <td>67.35</td>
        <td>69.31</td>
        <td>67.50</td>
        <td>66.62</td>
    </tr>
    <tr>
        <td>HalDet (LLAVA-13b)</td>
        <td>74.74</td>
        <td>76.68</td>
        <td>74.88</td>
        <td>74.34</td>
    </tr>
    <tr>
        <td>UniHD(GPT-4V)</td>
        <td>85.82</td>
        <td>85.83</td>
        <td>85.83</td>
        <td>85.82</td>
    </tr>
    <tr>
        <td>UniHD(GPT-4o)</td>
        <td>89.29</td>
        <td>89.28</td>
        <td>89.28</td>
        <td>89.28</td>
    </tr>
</table>

To view more detailed information about HalDet-LLaVA and the train dataset, please refer to the [readme](https://github.com/zjunlp/EasyDetect/blob/main/HalDet-LLaVA/README.md).

## 🔧Installation

**Installation for local development:**
```
conda create -n easydetect python=3.9.19
git clone https://github.com/zjunlp/EasyDetect.git
cd EasyDetect
pip install -r requirements.txt
```

**Installation for tools(GroundingDINO and MAERec):**
```
# install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cp -r GroundingDINO pipeline/GroundingDINO
cd pipeline/GroundingDINO/
pip install -e .
cd ..

# install MAERec
git clone https://github.com/Mountchicken/Union14M.git
cp -r Union14M/mmocr-dev-1.x pipeline/mmocr
cd pipeline/mmocr/
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
pip install timm
pip install -r requirements/albu.txt
pip install -r requirements.txt
pip install -v -e .
cd ..

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth -O dbnetpp.pth
wget https://github.com/Mountchicken/Union14M/releases/download/Checkpoint/maerec_b_union14m.pth -O maerec_b.pth
cd ..
```

---

## ⏩Quickstart

We provide example code for users to quickly get started with EasyDetect.

#### Step1: Write a configuration file in yaml format

Users can easily configure the parameters of EasyDetect in a yaml file or just quickly use the default parameters in the configuration file we provide. The path of the configuration file is EasyDetect/pipeline/config/config.yaml

```yaml
openai:
  api_key: Input your openai api key
  base_url: Input base_url, default is None
  temperature: 0.2  
  max_tokens: 1024
tool: 
  detect:
    groundingdino_config: the path of GroundingDINO_SwinT_OGC.py
    model_path: the path of groundingdino_swint_ogc.pth
    device: cuda:0
    BOX_TRESHOLD: 0.35
    TEXT_TRESHOLD: 0.25
    AREA_THRESHOLD: 0.001
  ocr:
    dbnetpp_config: the path of dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py
    dbnetpp_path: the path of dbnetpp.pth
    maerec_config: the path of maerec_b_union14m.py
    maerec_path: the path of maerec_b.pth
    device: cuda:0
    content: word.number
    cachefiles_path: the path of cache_files to save temp images
    BOX_TRESHOLD: 0.2
    TEXT_TRESHOLD: 0.25
  google_serper:
    serper_api_key: Input your serper api key
    snippet_cnt: 10
prompts:
  claim_generate: pipeline/prompts/claim_generate.yaml
  query_generate: pipeline/prompts/query_generate.yaml
  verify: pipeline/prompts/verify.yaml
```

#### Step2: Run with the Example Code
Example Code
```python
from pipeline.run_pipeline import *
pipeline = Pipeline()
text = "The cafe in the image is named \"Hauptbahnhof\""
image_path = "./examples/058214af21a03013.jpg"
type = "image-to-text"
response, claim_list = pipeline.run(text=text, image_path=image_path, type=type)
print(response)
print(claim_list)
```



---
## 🚩Citation

Please cite our repository if you use EasyDetect in your work.

```bibtex
@article{chen23factchd,
  author       = {Xiang Chen and Duanzheng Song and Honghao Gui and Chengxi Wang and Ningyu Zhang and 
                  Jiang Yong and Fei Huang and Chengfei Lv and Dan Zhang and Huajun Chen},
  title        = {FactCHD: Benchmarking Fact-Conflicting Hallucination Detection},
  journal      = {CoRR},
  volume       = {abs/2310.12086},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2310.12086},
  doi          = {10.48550/ARXIV.2310.12086},
  eprinttype    = {arXiv},
  eprint       = {2310.12086},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2310-12086.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@inproceedings{chen-etal-2024-unified-hallucination,
    title = "Unified Hallucination Detection for Multimodal Large Language Models",
    author = "Chen, Xiang  and
      Wang, Chenxi  and
      Xue, Yida  and
      Zhang, Ningyu  and
      Yang, Xiaoyan  and
      Li, Qiang  and
      Shen, Yue  and
      Liang, Lei  and
      Gu, Jinjie  and
      Chen, Huajun",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.178",
    pages = "3235--3252",
}
```

---



## 🎉Contributors

<a href="https://github.com/OpenKG-ORG/EasyDetect/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenKG-ORG/EasyDetect" />
</a>

We will offer long-term maintenance to fix bugs, solve issues and meet new requests. So if you have any problems, please put issues to us.
