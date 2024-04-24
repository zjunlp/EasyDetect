## Train dataset

the steps of train dataset construction:
- step1: We used LLaVA to generate raw response on the training sets of [MSCOCO-2014](https://cocodataset.org/), [VQA-v2](https://visualqa.org/) and [TextVQA](https://textvqa.org/dataset/).
- step2: We prompted GPT-3.5 to incorporate hallucinated text in terms of objects, attributes, scene text, and factual aspects, which were then manually reviewed.
- step3: We use the pipeline [UniHD](https://arxiv.org/abs/2402.03190) to generate labels and rationales for text, which were then subjected to manual screening and modifications to obtain the training dataset.

the train dataset metadata info:
We have constructed 1270 instructions for fine-tuning data. The ratio of hallucination claims to non-hallucination claims is 2244:1633. We provide reference tool information and reference prompt in train set. Below is an example of a train data:

```json
{
    "id": 6,
    "image_path": "/TextVQA/train_images/160dee3be9ec3cbc.jpg",
    "claim_list": ["The laptop brand is Toshiba","Toshiba is a multinational conglomerate with a rich history","Toshiba was founded in 1885"],
    "ref_tool_info": "Here is the object detection expert model's result: laptop [0.003, 0.001, 0.996, 0.996] \nHere is the scene text recognition expert model's result: ANNIVERSARY [0.065, 0.638, 0.952, 0.826] TONGFaNG [0.462, 0.523, 0.575, 0.542] \nHere is the external knowledge: 1. Toshiba Corporation (株式会社東芝, Kabushikigaisha Tōshiba, English: /təˈʃiːbə, tɒ-, toʊ-/) is a Japanese multinational electronics company headquartered in Minato, Tokyo, Japan. 2. Toshiba's early history has two strands: One is",
    "ref_claim_label": ["hallucination", "non-hallucination", "hallucination"],
    "ref_reason": [{"claim1": "hallucination","reason": "The scene text recognition expert model's result shows the text 'TONGFANG' on the laptop, not Toshiba. Therefore, there's a hallucination."},{"claim2": "non-hallucination","reason": "Based on the external knowledge provided, Toshiba is indeed a multinational conglomerate with a rich history. Therefore, there's no hallucination."},{"claim3": "hallucination","reason": "According to the external knowledge, Toshiba was founded in 1939 by the merger of Shibaura Seisakusho and Tokyo Denki, not in 1885. Therefore, there's a hallucination."}],
    "ref_prompt": "Given an image, a list of claims from Multimodal Large Language Models and some supplementary information by external tools, you are required to judge whether each claim in the list conflicts with the image, following these rules: \n1. You must carefully judge from four aspects, including the object, attributes, scene text and fact.\n2. You must carefully utilize supplementary information.\n3. You must carefully judge whether the visual information in the image conflicts with each claim. If there is a conflict, the result for that claim is labeled as 'hallucination'; otherwise, it is labeled as 'non-hallucination'.\n4. Finally, You MUST only respond in a dictionary format. DO NOT RESPOND WITH ANYTHING ELSE.\n"
}
```

**Note**:If you want to select LLaVA as the target model for training, you need to adjust to its [custom dataset format](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md).



## HalDet-LLaVA

HalDet-LLaVA is trained on the [MHaluBench training set](https://huggingface.co/datasets/openkg/MHaluBench/blob/main/MHaluBench_train.json) using LLaVA-v1.5, specific parameters can be found in the file [finetune_task_lora.sh](https://github.com/zjunlp/EasyDetect/blob/main/HalDet-LLaVA/finetune_task_lora.sh).

We trained HalDet-LLaVA on 1-A800 in 1 hour. If you don"t have enough GPU resources, we will soon provide model distributed training scripts.

You can inference our HalDet-LLaVA by using [inference.py](https://github.com/zjunlp/EasyDetect/blob/main/HalDet-LLaVA/inference.py)






