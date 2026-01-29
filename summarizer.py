#text summarization using a pre-trained model
from transformers import pipeline
from config import TEXT_SUMMARIZER

_summarizer = None # lazy global variable init so that we don't load model unless needed.

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # Auto-detect device (GPU if available, else CPU)
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device for text summarization: {'GPU' if device == 0 else 'CPU'}")
        _summarizer = pipeline(
            "text2text-generation",
            model=TEXT_SUMMARIZER,
            device=device
        )
    return _summarizer

def summarize_text(text: str):
    summarizer = get_summarizer()
    text = text.strip()
    if len(text) < 50: #if the original text is less than 50 characters hen no need to summarize cause it is already short
        return text
    result = summarizer(
        f"summarize: {text}",
        max_length=256, #not really need to define as it is already defined in the transformer model we are using 
        min_length=20, # same not really need to define
        do_sample=False, # for deterministic summary and no randomness
    )
    return result[0]["generated_text"]
