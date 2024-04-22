import argparse
import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import base64
import requests
from PIL import Image
from io import BytesIO
import re

class HalDetLLaVA:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, None, self.model_name, device="cuda:0")

    def image_parser(self, image_file, sep):
        out = image_file.split(sep)
        return out

    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out

    def get_response(self, query, image_file, conv_mode=None, sep=",", temperature=0.8, top_p=None, num_beams=1, max_new_tokens=2048):
        # Model
        disable_torch_init()
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv_mode = "llava_v1"
        if conv_mode is not None and conv_mode != conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, conv_mode, conv_mode
                )
            )
        else:
            conv_mode = conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_files = self.image_parser(image_file, sep)
        images = self.load_images(image_files)
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda(0)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
    
if __name__ == '__main__':
    model = HalDetLLaVA("zjunlp/HalDet-llava-7b")
    query = "Given an image, a list of claims from Multimodal Large Language Models and some supplementary information by external tools, you are required to judge whether each claim in the list conflicts with the image, following these rules: \n1. You must carefully judge from four aspects, including the object, attributes, scene text and fact.        \n2. You must carefully utilize supplementary information.  \n3. You must carefully judge whether the visual information in the image conflicts with each claim. If there is a conflict, the result for that claim is labeled as \"hallucination\"; otherwise, it is labeled as \"non-hallucination\".   \n4. Finally, You MUST only respond in a dictionary format. DO NOT RESPOND WITH ANYTHING ELSE.\nHere is the claim list: claim1: The cafe in the image is named \"Hauptbahnhof\" \nSupplementary information:\nHere is the object detection expert model's result: cafe [0.703, 0.621, 0.770, 0.650] \nHere is the scene text recognition expert model's result: Hauptbahnhof [0.571, 0.627, 0.622, 0.649] ZEITCAFE [0.707, 0.629, 0.775, 0.659] \nHere is the external knowledge: none information"
    image_file = "../examples/058214af21a03013.jpg"
    res = model.get_response(query,image_file)
    print(res)