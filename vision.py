# vision.py- image captioning using BLIP model
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional

# lazy load to speed startup means we only load the model when we first need it
_processor = None #prepares the image for the model
_model = None #produces the image captions

def _init_blip(model_name="Salesforce/blip-image-captioning-large", device=0): #loads it on the cuda if available 
    global _processor, _model # use global variables to load the model and processor only once
    if _processor is None:
        _processor = BlipProcessor.from_pretrained(model_name)
    if _model is None:
        _model = BlipForConditionalGeneration.from_pretrained(model_name).to(f"cuda:{device}" if device >= 0 else "cpu")
    return _processor, _model

def decode_b64_to_pil(b64: str) -> Image.Image: # convert base64 string back to PIL image, to be used as input to the BLIP model
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def summarize_image(image_b64: str, surrounding_text: Optional[str] = None, model_name=None, device=0) -> str:
    """
    Returns a combined caption: BLIP caption + optional surrounding context to improve usefulness.
    """
    model_name = model_name or "Salesforce/blip-image-captioning-large"
    processor, model = _init_blip(model_name=model_name, device=device)
    img = decode_b64_to_pil(image_b64)
    inputs = processor(images=img, return_tensors="pt").to(model.device)# the preprocessed image and optional text prompt , torch tensors ready for the model 
    outputs = model.generate(**inputs, max_length=40) # the raw model output - token ids 
    caption = processor.decode(outputs[0], skip_special_tokens=True) #readable decode text which stored as summary in the chroma DB
    if surrounding_text: #if the surrounding text is provided then we are giving the blip model both text
        #and image context to generate a better caption
        # combine succinctly
        return f"{caption}. Context: {surrounding_text.strip()[:300]}"
    return caption
