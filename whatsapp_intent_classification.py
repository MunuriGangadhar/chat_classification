import re
import json
import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Clean WhatsApp message
def preprocess_message(message):
    # Remove emojis, extra spaces, and normalize case
    message = re.sub(r'[^\w\s]', '', message)  # Remove emojis/punctuation
    message = re.sub(r'\s+', ' ', message).strip().lower()
    return message

# Classify intent and extract structured data
def classify_and_extract(message, model):
    prompt = f"""
Given a WhatsApp message, perform the following tasks:
1. Classify the Intent as one of: Contact Request, Information Query, Technical Support, Feedback, Purchase Inquiry, Booking Request, Order Status, Complaint, Casual Chat, Account Management, Event Inquiry, Subscription Inquiry, Location Inquiry, Recommendation Request, Policy Inquiry.
   If the message contains multiple intents, prioritize the primary intent and note secondary intents in the "notes" field.
2. Extract entities in a structured JSON format for the primary intent, as specified below. Return null for entities if not applicable.
3. Return the result as a JSON object with "intent", "entities", and "notes" keys wrapped in ```json```.

Intent Definitions and Entity Formats:
- Contact Request: {{"Location": "...", "Company": "..."}} 
  Example: "Contact Facebook in Bangalore" → ```json{{"intent": "Contact Request", "entities": {{"Location": "Bangalore", "Company": "Facebook"}}, "notes": null}}```
- Information Query: {{"Product": "..."}} 
  Example: "What are the features of the Apple Watch?" → ```json{{"intent": "Information Query", "entities": {{"Product": "Apple Watch"}}, "notes": null}}```
- Technical Support: {{"Issue": "..."}} 
  Example: "My app crashes" → ```json{{"intent": "Technical Support", "entities": {{"Issue": "app crash"}}, "notes": null}}```
- Feedback: {{"Sentiment": "positive/negative/mixed", "Topic": "..."}} 
  Example: "Great service!" → ```json{{"intent": "Feedback", "entities": {{"Sentiment": "positive", "Topic": "service"}}, "notes": null}}```
- Purchase Inquiry: {{"Product": "...", "Aspect": "price/availability/..."}} 
  Example: "How much is the premium plan?" → ```json{{"intent": "Purchase Inquiry", "entities": {{"Product": "premium plan", "Aspect": "price"}}, "notes": null}}```
- Booking Request: {{"Time": "...", "Quantity": "...", "Service": "..."}} 
  Example: "Book a table for two at 7 PM" → ```json{{"intent": "Booking Request", "entities": {{"Time": "7 PM", "Quantity": "two", "Service": "table"}}, "notes": null}}```
- Order Status: {{"OrderID": "..."}} 
  Example: "Where is my package?" → ```json{{"intent": "Order Status", "entities": {{"OrderID": null}}, "notes": null}}```
- Complaint: {{"Issue": "..."}} 
  Example: "The product is defective" → ```json{{"intent": "Complaint", "entities": {{"Issue": "defective product"}}, "notes": null}}```
- Casual Chat: None 
  Example: "Hey, how’s it going?" → ```json{{"intent": "Casual Chat", "entities": null, "notes": null}}```
- Account Management: {{"Task": "..."}} 
  Example: "I forgot my password" → ```json{{"intent": "Account Management", "entities": {{"Task": "password reset"}}, "notes": null}}```
- Event Inquiry: {{"Event": "...", "Aspect": "date/location/..."}} 
  Example: "When is the next webinar?" → ```json{{"intent": "Event Inquiry", "entities": {{"Event": "webinar", "Aspect": "date"}}, "notes": null}}```
- Subscription Inquiry: {{"Action": "...", "Service": "..."}} 
  Example: "How do I cancel my subscription?" → ```json{{"intent": "Subscription Inquiry", "entities": {{"Action": "cancel", "Service": "subscription"}}, "notes": null}}```
- Location Inquiry: {{"Location": "...", "Place": "..."}} 
  Example: "Where is your store in Mumbai?" → ```json{{"intent": "Location Inquiry", "entities": {{"Location": "Mumbai", "Place": "store"}}, "notes": null}}```
- Recommendation Request: {{"Category": "...", "Location": "..."}} 
  Example: "Recommend a restaurant in Delhi" → ```json{{"intent": "Recommendation Request", "entities": {{"Category": "restaurant", "Location": "Delhi"}}, "notes": null}}```
- Policy Inquiry: {{"Policy": "..."}} 
  Example: "What’s your refund policy?" → ```json{{"intent": "Policy Inquiry", "entities": {{"Policy": "refund"}}, "notes": null}}```

Message: {message}
Output:
"""
    
    # Generate response using Gemini
    response = model.generate_content(prompt).text
    
    # Extract JSON from response
    json_content = re.findall(r'```json(.*?)```', response, re.DOTALL)
    if json_content:
        json_content = json_content[0].strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_response": response}
    return {"error": "No JSON block found", "raw_response": response}

# Main function to process messages
def process_message(message, api_key):
    # Preprocess the message
    cleaned_message = preprocess_message(message)
    
    # Initialize Gemini model
    model = configure_gemini(api_key)
    
    # Classify intent and extract entities
    result = classify_and_extract(cleaned_message, model)
    return result

# Flask webhook for AutoResponder
@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid payload"}), 400
    
    message_text = data.get('message', '')
    from_phone_number = data.get('sender', '')
    
    if not message_text:
        return jsonify({"status": "Empty message"}), 200
    
    # Process message with Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Gemini API key not set"}), 500
    
    result = process_message(message_text, api_key)
    
    # Generate response
    response_text = f"Received intent: {result.get('intent')}"
    if result.get('entities'):
        response_text += f", Entities: {json.dumps(result['entities'])}"
    if result.get('notes'):
        response_text += f", Notes: {result['notes']}"
    
    # Return response for AutoResponder
    return jsonify({"response": response_text}), 200

# Example usage for offline testing
if __name__ == "__main__":
    # Get API key
    api_key ="AIzaSyAuJCRqZlmKR8A9zbmHE96_6Vs9MHAlER4" \
    ""
    
    # Sample messages
    messages = [
        "I need to contact Facebook in Bangalore",
        "What are the features of the Apple Watch?",
        "My app crashes when I open it",
        "Great service, but delivery was late",
        "How much is the premium plan?",
        "Book a table for two at 7 PM",
        "Where is my package?",
        "The product I received is defective",
        "Hey, how’s it going?",
        "I forgot my password",
        "When is the next webinar?",
        "How do I cancel my subscription?",
        "Where is your store in Mumbai?",
        "Recommend a restaurant in Delhi",
        "What’s your refund policy?",
        "Plz give me Google’s office number in Hydrabad 😊",
        "Tell me about iPhone 13 specs",
        "App keeps freezing on login screen, help!!",
        "Your customer support was awesome but the app is slow",
        "Is the basic plan available for purchase?",
        "Reserve a spot for 3 people tomorrow at 8pm",
        "Track my order #12345",
        "I got a broken phone, this is unacceptable!",
        "Yo, what’s up? Just chilling 😎",
        "Can u help me reset my account password?",
        "What’s the location of your next conference?",
        "Quiero cancelar mi suscripción, cómo lo hago?",
        "Where’s your outlet in Chennai city?",
        "Suggest a good coffee shop in Bangalore",
        "Can you explain your privacy policy?"
    ]

    # Offline testing
    print("Running offline test with sample messages...")
    for msg in messages:
        result = process_message(msg, api_key)
        print(f"Message: {msg}")
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 50)
    
    # Start Flask app
    print("Starting Flask webhook server...")
    app.run(host='0.0.0.0', port=5000)