# loader.py-PDF extraction using unstructured
import base64
from typing import List, Dict, Any
import warnings
from unstructured.partition.pdf import partition_pdf
from pypdf import PdfReader
from PIL import Image
import io
from config import PDF_FILEPATH


def safe_page_number(c):
    """Extract page_number safely from ElementMetadata.or if the 
    page_number is not there then it returns none and not let the pipeline break"""
    if hasattr(c, "metadata") and hasattr(c.metadata, "page_number"):
        return c.metadata.page_number
    return None

def load_pdf_text_table_elements(pdf_path: str = None) -> List[Dict[str, Any]]:
    """
    Use unstructured.partition.pdf to extract elements.
    Returns a list of dicts: {'type': 'text'|'table'|'image', 'content': str or base64, 'meta': {...}}
    """
    pdf_path = pdf_path or PDF_FILEPATH # if the caller passes a path use that other wise 
    # fall back to a default path
    chunks = partition_pdf(
        filename=pdf_path,
        strategy="fast",              
        infer_table_structure=False,  
        pdf_parser="pdfplumber",
        languages=["eng"],            
    )

    print(chunks)

    elements = []
    for c in chunks:
        t = str(type(c))
        page_num = safe_page_number(c)

        # Text elements
        if hasattr(c, "text") and c.text:
            #text = getattr(c, "text", None) or str(c)
            elements.append({
                "type": "text",
                "content": c.text,
                "meta": {"page_number": page_num}
            })

        # Table elements
        elif "Table" in t:
            elements.append({
                "type": "table",
                "content": str(c),
                "meta": {"page_number": page_num}
            })

    return elements

def pil_image_to_base64(img: Image.Image, format="PNG") -> str:
    """
    Convert a PIL Image to base64-encoded string.
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def _extract_images_from_xobject(xobj, page_num, elements):
    subtype = xobj.get("/Subtype")
    
    # Case 1: Direct Image
    if subtype == "/Image":
        try:
            img_data = xobj.get_data()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            img_b64 = pil_image_to_base64(img)
            
            elements.append({
                "type": "image",
                "content": img_b64, # PIL Image object for BLIP
                "meta": {"page_number": page_num}
            })
        except Exception:
            pass

    # Case 2: Form XObject → recurse
    elif subtype == "/Form":
        resources = xobj.get("/Resources", {})
        xobjects = resources.get("/XObject", {})
        for obj in xobjects.values():
            _extract_images_from_xobject(obj.get_object(), page_num, elements)

def load_pdf_image_elements(pdf_path: str = None) -> List[Dict[str, Any]]:
    """
    Docstring for load_pdf_image_elements
    
    :param pdf_path: Description
    :type pdf_path: str
    :return: Description
    :rtype: List[Dict[str, Any]]
    """
    reader = PdfReader(pdf_path or PDF_FILEPATH)
    elements = []

    for page_num, page in enumerate(reader.pages, start=1):
        resources = page.get("/Resources", {})
        xobjects = resources.get("/XObject", {})

        for xobj in xobjects.values():
            _extract_images_from_xobject(xobj.get_object(), page_num, elements)
    return elements

def load_pdf_elements(pdf_path: str = None) -> List[Dict[str, Any]]:
    pdf_path = pdf_path or PDF_FILEPATH # if the caller passes a path use that other wise

    elements = []
    elements.extend(load_pdf_text_table_elements(pdf_path))
    elements.extend(load_pdf_image_elements(pdf_path))

    return elements


if __name__ == "__main__":
    elems = load_pdf_elements()
    print(f"Extracted {len(elems)} elements from PDF.")