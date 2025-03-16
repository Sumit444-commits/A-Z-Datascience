import google.generativeai as genai
from google.colab import userdata
from PIL import Image
import io

# Get API Key from Colab's userdata (Replace with actual key name)
GOOGLE_API_KEY = userdata.get('AIzaSyD_dtdr6Qa_Q-DzO3z8e10S8Q9CrM60Ofs')  # Ensure this matches the stored variable name
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini Pro model
model = genai.GenerativeModel('gemini-pro')

# Load the image
img_path = 'image.png'  # Ensure this image exists in your working directory
img = Image.open(img_path)

# Convert image to bytes (needed for Gemini API)
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_bytes = img_byte_arr.getvalue()

# Generate response from the image
response = model.generate_content([img_bytes], stream=True)  # stream=True improves performance

# Print the response
for chunk in response:
    print(chunk.text)

# Embedding Example
result = genai.embed_content(
    model="embedding-001",  # Corrected model name
    content="What is the meaning of life?",
    task_type="retrieval_document",
    title="Embedding of single string"
)

print(result)
