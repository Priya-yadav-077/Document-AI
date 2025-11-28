# loader.py-PDF extraction using unstructured
import base64
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from config import PDF_FILEPATH

def safe_page_number(c):
    """Extract page_number safely from ElementMetadata.or if the 
    page_number is not there then it returns none and not let the pipeline break"""
    if hasattr(c, "metadata") and hasattr(c.metadata, "page_number"):
        return c.metadata.page_number
    return None

def load_pdf_elements(pdf_path: str = None) -> List[Dict[str, Any]]:
    """
    Use unstructured.partition.pdf to extract elements.
    Returns a list of dicts: {'type': 'text'|'table'|'image', 'content': str or base64, 'meta': {...}}
    """
    pdf_path = pdf_path or PDF_FILEPATH #if the caller passes a path use that other wise 
    #fall back to a default path
    chunks = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
        pdf_parser="pdfplumber"
    )

    elements = []
    for c in chunks:
        t = str(type(c))
        page_num = safe_page_number(c)

        # Text elements
        if "CompositeElement" in t or hasattr(c, "text"):
            text = getattr(c, "text", None) or str(c)
            elements.append({
                "type": "text",
                "content": text,
                "meta": {"page_number": page_num}
            })

        # Table elements
        elif "Table" in t:
            content = str(c)
            elements.append({
                "type": "table",
                "content": content,
                "meta": {"page_number": page_num}
            })

        # Image elements
        else:
            try:
                if hasattr(c.metadata, "image_base64"):
                    img_b64 = c.metadata.image_base64
                    elements.append({
                        "type": "image",
                        "content": img_b64,
                        "meta": {"page_number": page_num}
                    })
            except Exception:
                pass

    return elements
