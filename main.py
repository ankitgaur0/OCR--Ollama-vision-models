import ollama
from PIL import Image
import PyPDF2

import io
import base64
import os

# Function to convert image to base64 (required for Ollama vision models)
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Function to extract text from PDF (converts PDF pages to images internally if needed)
def pdf_to_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF text directly: {e}")
        return None

# Function to process input (image or PDF) with Ollama vision model
def extract_details(file_path):
    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in [".png", ".jpg", ".jpeg", ".bmp"]:
        # Process image
        try:
            # Convert image to base64
            img_base64 = image_to_base64(file_path)
            
            # Call Ollama vision model (assuming LLaVA is used)
            response = ollama.generate(
                model="llava",
                prompt="Extract all text and describe the details visible in this image.",
                images=[img_base64]
            )
            
            return response["response"]
        except Exception as e:
            return f"Error processing image: {e}"

    elif file_extension == ".pdf":
        # Process PDF
        try:
            # First attempt direct text extraction
            pdf_text = pdf_to_text(file_path)
            
            if pdf_text and pdf_text.strip():
                return pdf_text
            
            # If direct text extraction fails or is empty, convert to image and use vision model
            print("Falling back to vision model for PDF...")
            # For simplicity, we'll assume the PDF is single-page here
            # For multi-page PDFs, you'd need to convert each page to an image
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            
            if not images:
                return "No pages found in PDF."
            
            # Process the first page as an image
            temp_image_path = "temp_page.png"
            images[0].save(temp_image_path, "PNG")
            img_base64 = image_to_base64(temp_image_path)
            
            response = ollama.generate(
                model="llava",
                prompt="Extract all text and describe the details visible in this image.",
                images=[img_base64]
            )
            
            # Clean up temporary file
            os.remove(temp_image_path)
            return response["response"]
            
        except Exception as e:
            return f"Error processing PDF: {e}"
    else:
        return "Unsupported file format. Please provide a .pdf, .png, .jpg, .jpeg, or .bmp file."

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "/home/ankit/Desktop/OCR/bank paychecks.jpg"  # or "example.png"
    
    result = extract_details(file_path)
    print("Extracted Details:")
    print(result)