import ollama
import io
from PIL import Image
import base64

def encode_image(image_path):

        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
        
            # Save image to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # base64 encode
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64

# Initialize conversation history
conversation_history = []

# First message

conversation_history.append({
    'role': 'user',
    'content': 'What is strange about this image?',
    'images': [encode_image('tests/ollama-testing/scenescribe_board.jpg')],
})

response = ollama.chat(
    model='gemma3:4b',
    messages=conversation_history,
)

# Add assistant's response to history
assistant_response = response['message']['content']
conversation_history.append({
    'role': 'assistant',
    'content': assistant_response,
})

print("Assistant:", assistant_response)

# Follow-up question (without image this time)
conversation_history.append({
    'role': 'user',
    'content': 'Can you elaborate on the third point you mentioned?',
})

response = ollama.chat(
    model='gemma3:4b',
    messages=conversation_history,
)

print("Assistant:", response['message']['content'])