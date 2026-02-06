# loader.py - PDF extraction using unstructured
import base64
import os
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from config import PDF_FILEPATH

def safe_page_number(c):
    """Extract page_number safely from ElementMetadata."""
    if hasattr(c, "metadata") and hasattr(c.metadata, "page_number"):
        return c.metadata.page_number
    return None

def load_pdf_elements(pdf_path: str = None, use_alternate_loader: bool = False) -> List[Dict[str, Any]]:
    """
    Use unstructured.partition.pdf to extract elements.
    Returns a list of dicts: {'type': 'text'|'table'|'image', 'content': str or base64, 'meta': {...}}
    """
    pdf_path = pdf_path or PDF_FILEPATH
    print(f"Loading PDF elements from: {pdf_path}")
    
    # Validate PDF path exists
    if not pdf_path:
        raise FileNotFoundError(
            "No PDF file found. Please place a PDF file in the project directory or specify the path."
        )
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found at: {pdf_path}\n"
            f"Please check the file path and make sure the PDF exists."
        )
    
    if use_alternate_loader:
        print("Using alternate PDF loader...")
        from loader_alt import load_pdf_text_table_elements, load_pdf_image_elements
        text_table_elements = load_pdf_text_table_elements(pdf_path)
        image_elements = load_pdf_image_elements(pdf_path)
        print(f"Extracted {len(text_table_elements)} text/table elements and {len(image_elements)} image elements.")
        return text_table_elements + image_elements
    
    # Use hi_res strategy with pdfminer backend (more reliable than pdfplumber)
    chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",  # More accurate extraction
        infer_table_structure=True,
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
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
