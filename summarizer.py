#text summarization using a pre-trained model
from transformers import pipeline

TEXT_SUMMARIZER = "google/flan-t5-small"
_summarizer = None # lazy global variable init so that we don't load model unless needed.

def get_summarizer():
    global _summarizer
    if _summarizer is None: # checks if the model is already loaded 
        _summarizer = pipeline( # if not then load it using huggingface pipeline 
            "text2text-generation",
            model=TEXT_SUMMARIZER,
            device=0  # GPU
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
