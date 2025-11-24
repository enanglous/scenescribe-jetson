import ollama

# Initialize conversation history
conversation_history = []

# First message
with open('scenescribe_board.jpg', 'rb') as file:
    conversation_history.append({
        'role': 'user',
        'content': 'What is strange about this image?',
        'images': [file.read()],
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