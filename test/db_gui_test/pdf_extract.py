import fitz  # pymupdf
import chromadb
import requests
import io
from responses import HTTPStatus

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_chunks = [page.get_text("text") for page in doc]
    return text_chunks

def fetching_from_url(pdf_url):
    is_status_ok = False
    response = requests.get(pdf_url)
    if response.status_code == HTTPStatus.OK.value:
        pdf_bytes = response.content
        text_chunks = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        is_status_ok = True
    else:
        text_chunks = ""
    return text_chunks, is_status_ok
