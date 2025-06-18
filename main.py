import re
import json
import time
import google.generativeai as genai
from getpass import getpass
from pathlib import Path

def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# For removing punctuation and normalizing spaces
def preprocess_message(message):
    message = re.sub(r'[^\w\s]', '', message)
    message = re.sub(r'\s+', ' ', message).strip().lower()
    return message

def classify_and_extract(message, model):
    prompt = f"""
You are a smart assistant for understanding and extracting structured information from informal WhatsApp messages.

### Your Task:
Analyze the message below and:
1. **Classify the user's intent**.
2. **Extract structured entities** related to that intent.
3. Handle messages with informal language, spelling mistakes, emojis, abbreviations, or missing words.
4. If the intent is clear but some entity values are missing, set them to `null`.
5. If the message is irrelevant or doesn't match any listed intent, assign it to `Casual Chat` and set `entities` to `null`.

### Expected Output Format:
Respond only with a JSON block wrapped in triple backticks like this:

```json
{{
  "intent": "<Intent Name>",
  "entities": {{
    ...key-value pairs OR null...
  }}
}}

Message: {message}
Output:
"""
    try:
        response = model.generate_content(prompt).text
        json_content = re.findall(r'```json(.*?)```', response, re.DOTALL)
        if json_content:
            json_content = json_content[0].strip()
            return json.loads(json_content)
        return {"error": "No JSON block found", "raw_response": response}
    except Exception as e:
        return {"error": f"Model error: {str(e)}", "raw_response": None}

def process_message(message, model):
    cleaned_message = preprocess_message(message)
    return classify_and_extract(cleaned_message, model)

if __name__ == "__main__":
    api_key = getpass("Enter your Gemini API key: ")
    file_path = "/content/_chat.txt"  # Update this path as needed

    if not Path(file_path).is_file():
        print(f"File not found: {file_path}")
        exit()

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print("Processing messages...\n")
    model = configure_gemini(api_key)

    for line in lines:
        # Skip empty or irrelevant lines
        if not line.strip() or "omitted" in line or line.startswith("Near By Tech Parks"):
            continue

        # Extract message content after timestamp
        match = re.search(r"\]\s*(.*)", line)
        message = match.group(1) if match else line.strip()

        result = process_message(message, model)

        print(f"Original: {message}")
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 60)

        time.sleep(2)  # Delay of 2 seconds between API calls
import re
import json
import time
import google.generativeai as genai
from getpass import getpass
from pathlib import Path

def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# For removing punctuation and normalizing spaces
def preprocess_message(message):
    message = re.sub(r'[^\w\s]', '', message)
    message = re.sub(r'\s+', ' ', message).strip().lower()
    return message

def classify_and_extract(message, model):
    prompt = f"""
You are a smart assistant for understanding and extracting structured information from informal WhatsApp messages.

### Your Task:
Analyze the message below and:
1. **Classify the user's intent**.
2. **Extract structured entities** related to that intent.
3. Handle messages with informal language, spelling mistakes, emojis, abbreviations, or missing words.
4. If the intent is clear but some entity values are missing, set them to `null`.
5. If the message is irrelevant or doesn't match any listed intent, assign it to `Casual Chat` and set `entities` to `null`.

### Expected Output Format:
Respond only with a JSON block wrapped in triple backticks like this:

```json
{{
  "intent": "<Intent Name>",
  "entities": {{
    ...key-value pairs OR null...
  }}
}}

Message: {message}
Output:
"""
    try:
        response = model.generate_content(prompt).text
        json_content = re.findall(r'```json(.*?)```', response, re.DOTALL)
        if json_content:
            json_content = json_content[0].strip()
            return json.loads(json_content)
        return {"error": "No JSON block found", "raw_response": response}
    except Exception as e:
        return {"error": f"Model error: {str(e)}", "raw_response": None}

def process_message(message, model):
    cleaned_message = preprocess_message(message)
    return classify_and_extract(cleaned_message, model)

if __name__ == "__main__":
    api_key = getpass("Enter your Gemini API key: ")
    file_path = "/content/_chat.txt"  # Update this path as needed

    if not Path(file_path).is_file():
        print(f"File not found: {file_path}")
        exit()

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print("Processing messages...\n")
    model = configure_gemini(api_key)

    for line in lines:
        # Skip empty or irrelevant lines
        if not line.strip() or "omitted" in line or line.startswith("Near By Tech Parks"):
            continue

        # Extract message content after timestamp
        match = re.search(r"\]\s*(.*)", line)
        message = match.group(1) if match else line.strip()

        result = process_message(message, model)

        print(f"Original: {message}")
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 60)

        time.sleep(2)  # Delay of 2 seconds between API calls
