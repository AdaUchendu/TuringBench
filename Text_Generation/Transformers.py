import transformers
from tqdm.notebook import tqdm

""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import logging

import numpy as np
import torch
import pandas as pd
import parser

import keras




from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(_, tokenizer, prompt_text, temperature):
    if temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(model, tokenizer, prompt_text, temperature):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if xlm_language in available_languages:
            language = xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(_, tokenizer, prompt_text, temperature):
    padding_text = "hello"
    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(_, tokenizer, prompt_text, temperature):
    padding_text = "hello"
    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

    prompt_text = (padding_text if padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


import os
import json


def generate_texts(model_type=None, model_name_or_path=None, prompts=[""], length=20,
                   stop_token=None, temperature=1.0, repetition_penalty=1.0,
                   k=0, p=0.9, padding_text="", xlm_language="", seed=42,
                   num_return_sequences=1, dump_json_file=None
                   ):
    # parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    # args = parser.parse_args()

    device = torch.device("cuda")
    n_gpu = 1

    set_seed(seed, n_gpu)

    # Initialize the model and tokenizer
    try:
        model_type = model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)

    print('finished loading model.')

    # length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)
    # logger.info(args)

    print('i am after length')

    # loading processed titles
    has_processed = set()
    if dump_json_file is not None:
        if os.path.isfile(dump_json_file):
            with open(dump_json_file, 'r') as f:
                for line in f:
                    has_processed.add(json.loads(line)['Title'])

    print('i am after the dump file')
    generated_outputs = []
    for prompt in tqdm(prompts):
        # skip processed titles
        if prompt in has_processed:
            continue
        prompt_text = prompt  # args.prompt if args.prompt else input("Model prompt >>> ")

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(model_type)
            preprocessed_prompt_text = prepare_input(model, tokenizer, prompt_text, temperature)
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt",
                add_space_before_punct_symbol=True
            )
        else:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            min_length=300,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(stop_token) if stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                    prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)

        generated_outputs.append(generated_sequences)

        if dump_json_file is not None:
            with open(dump_json_file, 'a') as f:
                f.write(json.dumps({'Title': prompt, 'Generated_output': generated_sequences}) + '\n')

    return generated_outputs


# if __name__ == '__main__':
#
#     #READ DATA INTO FILE
#     #ata = pd.read_csv()
#     #headlines = data['Article']
#
#     if opt.generator == 'ctrl':
#         prompts = ["News " + headline for headline in headlines]
#
#     generated = generate_texts(model_type=opt.generator, model_name_or_path=opt.generator, prompts=prompts,
#                                length=500, stop_token=None, temperature=0.4, repetition_penalty=2.2,
#                                k=0, p=0.9, padding_text="", xlm_language="", seed=42,
#                                num_return_sequences=1, dump_json_file='./gen/ctrl_generation.json')