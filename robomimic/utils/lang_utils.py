import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection

os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
lang_emb_model = None
tz = None

LANG_EMB_OBS_KEY = "lang_emb"

def _get_cache_dir():
    return os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))

def _get_lang_emb_model():
    global lang_emb_model
    if lang_emb_model is None:
        lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            tokenizer,
            cache_dir=_get_cache_dir(),
        ).eval()
    return lang_emb_model

def _get_tokenizer():
    global tz
    if tz is None:
        tz = AutoTokenizer.from_pretrained(
            tokenizer,
            TOKENIZERS_PARALLELISM=True,
            cache_dir=_get_cache_dir(),
        )
    return tz

def get_lang_emb(lang):
    if lang is None:
        return None
    
    tokens = _get_tokenizer()(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    lang_emb = _get_lang_emb_model()(**tokens)['text_embeds'].detach()[0]

    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)
