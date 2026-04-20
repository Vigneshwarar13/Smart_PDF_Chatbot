# Helper functions

def validate_api_key(api_key):
    """Validate if API key is provided."""
    if not api_key:
        raise ValueError("API key is required.")

def validate_pdf_docs(pdf_docs):
    """Validate if PDF documents are uploaded."""
    if not pdf_docs:
        raise ValueError("Please upload PDF documents.")