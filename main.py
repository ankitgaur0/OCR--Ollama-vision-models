import ollama
from PIL import Image
import PyPDF2
import io
import base64
import os
import json

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

# Function to process input (image or PDF) with Ollama vision model and extract key-value pairs
def extract_key_value_pairs(file_path):
    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in [".png", ".jpg", ".jpeg", ".bmp"]:
        # Process image
        try:
            # Convert image to base64
            img_base64 = image_to_base64(file_path)
            
            # Call Ollama vision model with a prompt to extract key-value pairs
            response = ollama.generate(
                model="llava",
                prompt="Analyze this image of a document (e.g., bank statement, paycheck, or form) and extract all key-value pairs. Return the result as a JSON object where keys are the field names (e.g., 'NAME', 'ACCOUNT NO.', 'DATE') and values are their corresponding data. Handle various formats and ensure accuracy. If a field is unclear or missing, include it with a value of 'N/A'.",
                images=[img_base64]
            )
            
            # Try to parse the response as JSON
            try:
                result = json.loads(response["response"])
                return result
            except json.JSONDecodeError:
                # If the response isn't valid JSON, parse it manually or return as text
                print("Warning: Response is not valid JSON. Attempting to parse manually.")
                return parse_text_response(response["response"])
                
        except Exception as e:
            return {"error": f"Error processing image: {e}"}

    elif file_extension == ".pdf":
        # Process PDF
        try:
            # First attempt direct text extraction
            pdf_text = pdf_to_text(file_path)
            
            if pdf_text and pdf_text.strip():
                # Use the text directly within the prompt (combine with the prompt string)
                prompt_with_text = f"""
                Analyze this text from a PDF document (e.g., bank statement, paycheck, or form) and extract all key-value pairs. 
                Return the result as a JSON object where keys are the field names (e.g., 'NAME', 'ACCOUNT NO.', 'DATE') and values are their corresponding data. 
                Handle various formats and ensure accuracy. If a field is unclear or missing, include it with a value of 'N/A'.
                
                Text content: {pdf_text}
                """
                
                response = ollama.generate(
                    model="llava",
                    prompt=prompt_with_text
                )
                
                try:
                    result = json.loads(response["response"])
                    return result
                except json.JSONDecodeError:
                    print("Warning: Response is not valid JSON. Attempting to parse manually.")
                    return parse_text_response(response["response"])
            
            # If direct text extraction fails or is empty, convert to image and use vision model
            print("Falling back to vision model for PDF...")
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            
            if not images:
                return {"error": "No pages found in PDF."}
            
            # Process the first page as an image
            temp_image_path = "temp_page.png"
            images[0].save(temp_image_path, "PNG")
            img_base64 = image_to_base64(temp_image_path)
            
            response = ollama.generate(
                model="llava",
                prompt="Analyze this image of a PDF document (e.g., bank statement, paycheck, or form) and extract all key-value pairs. Return the result as a JSON object where keys are the field names (e.g., 'NAME', 'ACCOUNT NO.', 'DATE') and values are their corresponding data. Handle various formats and ensure accuracy. If a field is unclear or missing, include it with a value of 'N/A'.",
                images=[img_base64]
            )
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            try:
                result = json.loads(response["response"])
                return result
            except json.JSONDecodeError:
                print("Warning: Response is not valid JSON. Attempting to parse manually.")
                return parse_text_response(response["response"])
                
        except Exception as e:
            return {"error": f"Error processing PDF: {e}"}
    else:
        return {"error": "Unsupported file format. Please provide a .pdf, .png, .jpg, .jpeg, or .bmp file."}

# Helper function to manually parse text response into key-value pairs
def parse_text_response(text_response):
    key_value_pairs = {}
    lines = text_response.split("\n")
    current_key = None
    
    for line in lines:
        line = line.strip()
        if ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
            if key and value:
                key_value_pairs[key] = value
        elif current_key and line:
            # Append to the value of the last key if it's a continuation
            key_value_pairs[current_key] = (key_value_pairs[current_key] + " " + line).strip()
    
    # Handle cases where keys or values might be missing or unclear
    if not key_value_pairs:
        key_value_pairs["error"] = "Unable to parse key-value pairs from response."
    
    return key_value_pairs

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "/home/ankit/Desktop/OCR/pdf2.pdf"  # Update with your image or PDF path
    
    result = extract_key_value_pairs(file_path)
    print(type(result))
    
    # Print the result in a formatted way
    print("Extracted Key-Value Pairs:")
    if isinstance(result, dict):
        import json
        print(json.dumps(result, indent=2))
    else:
        print(result)