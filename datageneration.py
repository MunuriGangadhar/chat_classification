import json
import pandas as pd
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import time
import logging
import re
import random
import backoff
import signal
import sys
from google.api_core.exceptions import GoogleAPIError, Unauthenticated

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# def build_prompt(batch_size, batch_num, seed):
#     """
#     Build a dynamic prompt with randomness, variety hints, and all provided examples to reduce duplicates.
#     """
#     return f"""
# You are a creative assistant tasked with generating realistic WhatsApp group chat messages in English, simulating informal, lively, and diverse conversations typical of group chats in India. The messages should feel natural, casual, and varied in tone, style, and content, resembling real-world queries. Each message must fall into one of three intent categories: `Person`, `Product`, or `Place`. You will also extract structured entities relevant to the intent.

# ### Your Task:
# 1. Generate **{batch_size} unique** WhatsApp-style messages in English, ensuring no duplicates.
# 2. Classify each message's intent as `Person`, `Product`, or `Place`.
# 3. Extract structured entities as key-value pairs relevant to the intent, focusing on details like location, product type, price, roles, skills, or purposes.
# 4. Ensure messages are informal, varied, and mimic real-world group chat scenarios, including colloquial language, abbreviations, slang, emojis, or casual phrasing (e.g., "guys", "pls", "any leads?", "what’s good?").
# 5. If some entity values are missing, set them to `null`.
# 6. Exclude purely conversational messages (e.g., "Hey, how’s it going?") or intents outside `Person`, `Product`, or `Place`.

# ### Guidelines for Diversity:
# - **Person**: Messages seeking professionals (e.g., tutors, designers, developers) with specific skills, locations, or contexts (e.g., events, projects). Use varied phrasing like "any leads on", "need someone ASAP", "who’s good at", or "know anyone who".
# - **Product**: Messages inquiring about products (e.g., phones, gadgets, appliances) with details like price, brand, features, or use cases. Include queries like "what’s the best deal", "any recos", "worth it?", or "where to buy".
# - **Place**: Messages asking for recommendations about locations (e.g., restaurants, cafes, parks) with specific purposes or characteristics. Use phrases like "where to chill", "good spot for", "any hidden gems", or "best place for".
# - Vary locations (e.g., Indian cities like Hyderabad, Mumbai, Delhi, or areas like Indranagar, Bandra, Koramangala), contexts, and details to ensure wide coverage.
# - Use different sentence structures, tones (e.g., urgent, curious, chill).
# - Reflect Indian group chat culture: include local references (e.g., "near MG Road", "for Diwali", "Navratri event"), casual abbreviations (e.g., "pls", "k"), and diverse scenarios (e.g., events, work, leisure, festivals).
# - **Randomization**: Use seed {seed} to make this generation different from previous ones. Avoid repeating messages or styles from earlier batches. Include fresh topics, styles, and contexts for batch {batch_num}.

# ### Expected Output Format:
# Return a JSON array of objects, each containing:
# ```json
# {{
# "original_message": "<Generated Message>",
# "intent": "<Person|Product|Place>",
# "entities": {{ ...key-value pairs or null... }}
# }}
# ```

# ### Examples:
# - {{"original_message": "Anyone know a good web developer in Hyderabad for a startup project?", "intent": "Person", "entities": {{"Skill": "Web Development", "Location": "Hyderabad", "Tags": ["Startup"]}}}}
# - {{"original_message": "Need a dance choreographer in Chennai for a wedding event next month", "intent": "Person", "entities": {{"Skill": "Dance Choreographer", "Location": "Chennai", "Event": "Wedding", "Timeframe": "Next Month"}}}}
# - {{"original_message": "Looking for a math tutor in Mumbai who can teach 12th-grade CBSE online", "intent": "Person", "entities": {{"Role": "Math Tutor", "Location": "Mumbai", "TargetAudience": "12th-grade CBSE", "Mode": "Online"}}}}
# - {{"original_message": "Need a UX researcher in Bangalore with experience in mobile apps", "intent": "Person", "entities": {{"Skill": "UX Researcher", "Location": "Bangalore", "Expertise": "Mobile Apps", "Experience": "Any"}}}}
# - {{"original_message": "Any experienced interior designer in Delhi for home renovations?", "intent": "Person", "entities": {{"Skill": "Interior Design", "Location": "Delhi", "Tags": ["Home Renovations"]}}}}
# - {{"original_message": "Know a machine learning expert in Patna for a 6-month contract?", "intent": "Person", "entities": {{"Skill": "Machine Learning", "Location": "Patna", "Tags": ["Contract", "6-month"]}}}}
# - {{"original_message": "Seeking a digital marketing specialist in Vizag with SEO expertise", "intent": "Person", "entities": {{"Role": "Digital Marketing Specialist", "Location": "Vizag", "Expertise": "SEO", "Availability": null}}}}
# - {{"original_message": "Searching for a physics tutor in Gurgaon for high school IB students", "intent": "Person", "entities": {{"Role": "Physics Tutor", "Location": "Gurgaon", "TargetAudience": "High School IB", "Mode": "Hybrid"}}}}
# - {{"original_message": "Hunting for a freelance animator in Chennai for short ad films", "intent": "Person", "entities": {{"Role": "Animator", "Location": "Chennai", "ProjectType": "Ad Films", "EmploymentType": "Freelance"}}}}
# - {{"original_message": "Find me a history tutor in Mumbai for IGCSE curriculum", "intent": "Person", "entities": {{"Role": "History Tutor", "Location": "Mumbai", "Tags": ["IGCSE Students", "Online"]}}}}
# - {{"original_message": "What’s the best Samsung mobile under 10K?", "intent": "Product", "entities": {{"Product": "Mobile", "Price": "Rs. 10000", "Brand": "Samsung", "Details": null}}}}
# - {{"original_message": "Suggest a phone under 15k with a good camera and battery life.", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Tags": ["Camera", "Battery Life"]}}}}
# - {{"original_message": "Which Samsung phone has the best camera for vlogging?", "intent": "Product", "entities": {{"Product": "Phone", "Brand": "Samsung", "Feature": "Camera", "UseCase": "Vlogging"}}}}
# - {{"original_message": "Can anyone suggest a gaming phone under 15k with a high refresh rate?", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Features": ["High Refresh Rate", "Gaming"]}}}}
# - {{"original_message": "Searching for a projector for classroom presentations?", "intent": "Product", "entities": {{"Product": "Projector", "UseCase": "Classroom Presentations", "Location": null}}}}
# - {{"original_message": "What’s a good fitness tracker under 4k for running in Pune?", "intent": "Product", "entities": {{"Product": "Fitness Tracker", "Price": "Rs. 4000", "Purpose": "Running", "Location": "Pune"}}}}
# - {{"original_message": "Can anyone recommend a smart doorbell under 8k with Wi-Fi connectivity?", "intent": "Product", "entities": {{"Product": "Smart Doorbell", "Price": "Rs. 8000", "Connectivity": "Wi-Fi"}}}}
# - {{"original_message": "Searching for a Xiaomi phone with 5G in Hyderabad?", "intent": "Product", "entities": {{"Product": "Phone", "Brand": "Xiaomi", "Connectivity": "5G", "Location": "Hyderabad"}}}}
# - {{"original_message": "Guys, what’s a good smartwatch under 6k with heart rate monitoring?", "intent": "Product", "entities": {{"Product": "Smartwatch", "Price": "Rs. 6000", "Feature": "Heart Rate Monitoring"}}}}
# - {{"original_message": "Searching for a water purifier for home use in Delhi?", "intent": "Product", "entities": {{"Product": "Water Purifier", "UseCase": "Home Use", "Location": "Delhi"}}}}
# - {{"original_message": "Anyone got ideas for a drone under 30k with 4K camera?", "intent": "Product", "entities": {{"Product": "Drone", "Price": "Rs. 30000", "Feature": "4K Camera"}}}}
# - {{"original_message": "Got any leads on a coffee grinder under 8k for espresso?", "intent": "Product", "entities": {{"Product": "Coffee Grinder", "Price": "Rs. 8000", "Tags": ["Espresso"]}}}}
# - {{"original_message": "What’s the best Biryani place in Indranagar?", "intent": "Place", "entities": {{"Location": "Indranagar, Bangalore", "Type": "Restaurant", "Tags": ["Biryani"]}}}}
# - {{"original_message": "Any nice cafes in Bandra West where I can work?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Bandra West", "Purpose": "Work"}}}}
# - {{"original_message": "Can anyone share a peaceful park in Chennai for morning walks?", "intent": "Place", "entities": {{"Type": "Park", "Location": "Chennai", "Purpose": "Morning Walks"}}}}
# - {{"original_message": "Searching for a quiet tea house in Pune for afternoon chats?", "intent": "Place", "entities": {{"Type": "Tea House", "Location": "Pune", "Purpose": "Afternoon Chats", "Ambiance": "Quiet"}}}}
# - {{"original_message": "Can you guys point to a karaoke lounge in Hyderabad under 2k for parties?", "intent": "Place", "entities": {{"Type": "Karaoke Lounge", "Location": "Hyderabad", "Price": "Rs. 2000", "Purpose": "Parties"}}}}
# - {{"original_message": "Know a nice juice bar in Delhi for fresh drinks?", "intent": "Place", "entities": {{"Type": "Juice Bar", "Location": "Delhi", "Tags": ["Fresh Drinks"]}}}}
# - {{"original_message": "Working from Andheri tomorrow, any laptop-friendly coffee spots?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Andheri", "Purpose": "Remote Work"}}}}
# - {{"original_message": "Where do you guys buy plants in Mumbai? Any aesthetic nursery?", "intent": "Place", "entities": {{"Type": "Plant Nursery", "Location": "Mumbai", "Tags": ["Aesthetic"]}}}}
# - {{"original_message": "Need midnight maggi places near MG Road", "intent": "Place", "entities": {{"Type": "Late Night Eatery", "Location": "MG Road", "Tags": ["Maggi", "Midnight Food"]}}}}
# - {{"original_message": "Best salon for men’s haircut in Rajajinagar?", "intent": "Place", "entities": {{"Type": "Salon", "Location": "Rajajinagar", "Tags": ["Men", "Haircut"]}}}}

# ### Output:
# Generate exactly {batch_size} unique messages in a single JSON array:
# ```json
# [
# {{"original_message": "...", "intent": "...", "entities": {{...}}}},
# ...
# ]
# ```
# """


def build_prompt(batch_size, batch_num, seed):
    """
    Build a dynamic prompt with randomness, variety hints, and examples to generate realistic WhatsApp group chat messages.
    """
    return f"""
You are a creative assistant tasked with generating realistic WhatsApp group chat messages in English, simulating informal, lively, and diverse conversations typical of group chats in India. The messages should feel natural, casual, and varied in tone, style, and content, resembling real-world queries. Each message must fall into one of three intent categories: `Person`, `Product`, or `Place`. You will also extract structured entities relevant to the intent.

### Your Task:
1. Generate **{batch_size} unique** WhatsApp-style messages in English, ensuring no duplicates.
2. Classify each message's intent as `Person`, `Product`, or `Place`.
3. Extract structured entities as key-value pairs relevant to the intent, focusing on details like location, product type, price, roles, skills, or purposes.
4. Ensure messages are informal, varied, and mimic real-world group chat scenarios, including colloquial language, abbreviations, slang, or casual phrasing (e.g., "guys", "pls", "any leads?", "what’s good?", "bhai", "yaar"). Do not include emojis.
5. If some entity values are missing, set them to `null`.
6. Exclude purely conversational messages (e.g., "Hey, how’s it going?") or intents outside `Person`, `Product`, or `Place`.

### Guidelines for Diversity:
- **Person**: Messages seeking professionals (e.g., tutors, designers, developers) with specific skills, locations, or contexts (e.g., events, projects). Use varied phrasing like "any leads on", "need someone ASAP", "who’s good at", "know anyone who", or "bhai, koi hai?".
- **Product**: Messages inquiring about products (e.g., phones, gadgets, appliances) with details like price, brand, features, or use cases. Include queries like "what’s the best deal", "any recos", "worth it?", "kahan milega?", or "koi sasta option?".
- **Place**: Messages asking for recommendations about locations (e.g., restaurants, cafes, parks) with specific purposes or characteristics. Use phrases like "where to chill", "good spot for", "any hidden gems", "koi mast jagah?", or "best place for".
- Vary locations (e.g., Indian cities like Hyderabad, Mumbai, Delhi, or areas like Indiranagar, Bandra, Koramangala), contexts, and details to ensure wide coverage.
- Use different sentence structures, tones (e.g., urgent, curious, chill, excited).
- Reflect Indian group chat culture: include local references (e.g., "near MG Road", "for Diwali", "Navratri vibes"), casual abbreviations (e.g., "pls", "k", "bro"), and diverse scenarios (e.g., events, work, leisure, festivals, food cravings).
- **Randomization**: Use seed {seed} to make this generation different from previous ones. Avoid repeating messages or styles from earlier batches. Include fresh topics, styles, and contexts for batch {batch_num}.

### Expected Output Format:
Return a JSON array of objects, each containing:
```json
{{
"original_message": "<Generated Message>",
"intent": "<Person|Product|Place>",
"entities": {{ ...key-value pairs or null... }}
}}
```

### Examples:
- {{"original_message": "Yo guys, any leads on a solid web developer in Hyderabad for our startup? Need ASAP", "intent": "Person", "entities": {{"Skill": "Web Development", "Location": "Hyderabad", "Tags": ["Startup"], "Urgency": "ASAP"}}}}
- {{"original_message": "Bhai, need a dance choreographer in Chennai for sis’s wedding sangeet next month", "intent": "Person", "entities": {{"Skill": "Dance Choreographer", "Location": "Chennai", "Event": "Wedding Sangeet", "Timeframe": "Next Month"}}}}
- {{"original_message": "Anyone know a math tutor in Mumbai for 12th CBSE? Online classes preferred, yaar", "intent": "Person", "entities": {{"Role": "Math Tutor", "Location": "Mumbai", "TargetAudience": "12th-grade CBSE", "Mode": "Online"}}}}
- {{"original_message": "Need a UX designer in Bangalore who’s worked on mobile apps. Any recos?", "intent": "Person", "entities": {{"Skill": "UX Designer", "Location": "Bangalore", "Expertise": "Mobile Apps", "Experience": "Any"}}}}
- {{"original_message": "Guys, any interior designer in Delhi for home makeover? Budget-friendly pls", "intent": "Person", "entities": {{"Skill": "Interior Design", "Location": "Delhi", "Tags": ["Home Makeover", "Budget-Friendly"]}}}}
- {{"original_message": "Koi machine learning pro in Patna for a short-term gig? 6 months max", "intent": "Person", "entities": {{"Skill": "Machine Learning", "Location": "Patna", "Tags": ["Contract", "6-month"]}}}}
- {{"original_message": "Need a digital marketing guru in Vizag who’s a pro at SEO. Any leads?", "intent": "Person", "entities": {{"Role": "Digital Marketing Specialist", "Location": "Vizag", "Expertise": "SEO", "Availability": null}}}}
- {{"original_message": "Yaar, any physics tutor in Gurgaon for IB kids? Hybrid classes cool?", "intent": "Person", "entities": {{"Role": "Physics Tutor", "Location": "Gurgaon", "TargetAudience": "High School IB", "Mode": "Hybrid"}}}}
- {{"original_message": "Hunting for a freelance animator in Chennai for some dope ad films", "intent": "Person", "entities": {{"Role": "Animator", "Location": "Chennai", "ProjectType": "Ad Films", "EmploymentType": "Freelance"}}}}
- {{"original_message": "Bro, need a history tutor in Mumbai for IGCSE. Online only, pls", "intent": "Person", "entities": {{"Role": "History Tutor", "Location": "Mumbai", "Tags": ["IGCSE Students", "Online"]}}}}
- {{"original_message": "What’s the best Samsung phone under 10k? Need a budget beast", "intent": "Product", "entities": {{"Product": "Mobile", "Price": "Rs. 10000", "Brand": "Samsung", "Details": null}}}}
- {{"original_message": "Guys, suggest a phone under 15k with killer camera and battery. Kahan milega?", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Tags": ["Camera", "Battery Life"]}}}}
- {{"original_message": "Which Samsung phone’s camera is best for vlogging? Need pro-level stuff", "intent": "Product", "entities": {{"Product": "Phone", "Brand": "Samsung", "Feature": "Camera", "UseCase": "Vlogging"}}}}
- {{"original_message": "Any gaming phone under 15k with high refresh rate? PubG vibes", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Features": ["High Refresh Rate", "Gaming"]}}}}
- {{"original_message": "Need a projector for classroom talks. Any budget options?", "intent": "Product", "entities": {{"Product": "Projector", "UseCase": "Classroom Presentations", "Location": null}}}}
- {{"original_message": "Yo, what’s a good fitness tracker under 4k for running in Pune?", "intent": "Product", "entities": {{"Product": "Fitness Tracker", "Price": "Rs. 4000", "Purpose": "Running", "Location": "Pune"}}}}
- {{"original_message": "Anyone know a smart doorbell under 8k with Wi-Fi? Home security upgrade", "intent": "Product", "entities": {{"Product": "Smart Doorbell", "Price": "Rs. 8000", "Connectivity": "Wi-Fi"}}}}
- {{"original_message": "Looking for a 5G Xiaomi phone in Hyderabad. Any deals?", "intent": "Product", "entities": {{"Product": "Phone", "Brand": "Xiaomi", "Connectivity": "5G", "Location": "Hyderabad"}}}}
- {{"original_message": "Bhai, suggest a smartwatch under 6k with heart rate tracking. Mast hona chahiye", "intent": "Product", "entities": {{"Product": "Smartwatch", "Price": "Rs. 6000", "Feature": "Heart Rate Monitoring"}}}}
- {{"original_message": "Need a water purifier for home in Delhi. Koi solid option?", "intent": "Product", "entities": {{"Product": "Water Purifier", "UseCase": "Home Use", "Location": "Delhi"}}}}
- {{"original_message": "Any dope drones under 30k with 4K camera? For vlogs and travel", "intent": "Product", "entities": {{"Product": "Drone", "Price": "Rs. 30000", "Feature": "4K Camera", "Tags": ["Vlogging", "Travel"]}}}}
- {{"original_message": "Yaar, any coffee grinder under 8k for espresso? Coffee lover here", "intent": "Product", "entities": {{"Product": "Coffee Grinder", "Price": "Rs. 8000", "Tags": ["Espresso"]}}}}
- {{"original_message": "Where’s the best biryani in Indiranagar? Craving some spice", "intent": "Place", "entities": {{"Location": "Indiranagar, Bangalore", "Type": "Restaurant", "Tags": ["Biryani"]}}}}
- {{"original_message": "Any chill cafes in Bandra West for work? Need Wi-Fi and coffee", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Bandra West", "Purpose": "Work", "Features": ["Wi-Fi"]}}}}
- {{"original_message": "Guys, any peaceful park in Chennai for morning jogs?", "intent": "Place", "entities": {{"Type": "Park", "Location": "Chennai", "Purpose": "Morning Jogs"}}}}
- {{"original_message": "Looking for a cozy tea house in Pune for chill evening chats. Suggestions?", "intent": "Place", "entities": {{"Type": "Tea House", "Location": "Pune", "Purpose": "Evening Chats", "Ambiance": "Cozy"}}}}
- {{"original_message": "Koi mast karaoke lounge in Hyderabad under 2k for a bday bash?", "intent": "Place", "entities": {{"Type": "Karaoke Lounge", "Location": "Hyderabad", "Price": "Rs. 2000", "Purpose": "Birthday Party"}}}}
- {{"original_message": "Need a juice bar in Delhi for fresh smoothies. Any hidden gems?", "intent": "Place", "entities": {{"Type": "Juice Bar", "Location": "Delhi", "Tags": ["Fresh Smoothies"]}}}}
- {{"original_message": "In Andheri tomorrow, any laptop-friendly cafes for WFH?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Andheri", "Purpose": "Remote Work"}}}}
- {{"original_message": "Where to buy cool plants in Mumbai? Need aesthetic vibes for home", "intent": "Place", "entities": {{"Type": "Plant Nursery", "Location": "Mumbai", "Tags": ["Aesthetic"]}}}}
- {{"original_message": "Craving midnight maggi near MG Road. Any 24/7 spots?", "intent": "Place", "entities": {{"Type": "Late Night Eatery", "Location": "MG Road", "Tags": ["Maggi", "Midnight Food"]}}}}
- {{"original_message": "Best salon in Rajajinagar for men’s haircut? Need a fresh look", "intent": "Place", "entities": {{"Type": "Salon", "Location": "Rajajinagar", "Tags": ["Men", "Haircut"]}}}}
- {{"original_message": "Guys, need a caterer in Bangalore for a small Diwali party. Any recos?", "intent": "Person", "entities": {{"Role": "Caterer", "Location": "Bangalore", "Event": "Diwali Party", "Scale": "Small"}}}}
- {{"original_message": "Any budget earbuds under 2k for gym use? Need sweat-proof ones", "intent": "Product", "entities": {{"Product": "Earbuds", "Price": "Rs. 2000", "Tags": ["Gym", "Sweat-Proof"]}}}}
- {{"original_message": "Koi acha rooftop bar in Kolkata for Navratri nights?", "intent": "Place", "entities": {{"Type": "Rooftop Bar", "Location": "Kolkata", "Purpose": "Navratri Nights"}}}}
- {{"original_message": "Need a yoga instructor in Jaipur for group sessions. Any leads?", "intent": "Person", "entities": {{"Role": "Yoga Instructor", "Location": "Jaipur", "Tags": ["Group Sessions"]}}}}
- {{"original_message": "Looking for a backpack under 3k for college. Durable wala chahiye", "intent": "Product", "entities": {{"Product": "Backpack", "Price": "Rs. 3000", "UseCase": "College", "Feature": "Durable"}}}}
- {{"original_message": "Any pet-friendly cafes in Gurgaon for a chill Sunday?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Gurgaon", "Purpose": "Chill Sunday", "Tags": ["Pet-Friendly"]}}}}

### Output:
Generate exactly {batch_size} unique messages in a single JSON array:
```json
[
{{"original_message": "...", "intent": "...", "entities": {{...}}}},
...
]
```
"""

def configure_vertex_ai(project_id, location, model_name, all_results):
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)
        logger.info(f"Vertex AI Gemini model initialized: {model_name}")
        return model
    except Unauthenticated as e:
        logger.error(f"Authentication failed: {str(e)}. Saving any existing results and exiting.")
        if all_results:
            save_results(all_results)
        print(f"Authentication failed: {str(e)}. Please authenticate with Google Cloud CLI using 'gcloud auth login'.")
        sys.exit(1)
    except GoogleAPIError as e:
        logger.error(f"Google API error: {str(e)}. Saving any existing results and exiting.")
        if all_results:
            save_results(all_results)
        print(f"Google API error: {str(e)}. Please check your credentials or network connection.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}. Saving any existing results and exiting.")
        if all_results:
            save_results(all_results)
        print(f"Failed to initialize Vertex AI client: {str(e)}.")
        sys.exit(1)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_whatsapp_messages_batch(batch_size, batch_num, model):
    """
    Generate a batch of WhatsApp messages with retry logic.
    """
    seed = random.randint(1, 999999)
    prompt = build_prompt(batch_size, batch_num, seed)
    generated_messages = set()
    results_list = []

    try:
        logger.debug(f"Sending request to Vertex AI for batch {batch_num} with seed {seed}")
        response = model.generate_content(prompt)
        content = response.text.strip()
        logger.debug(f"Raw response content for batch {batch_num}: {content}")

        # Extract JSON from response
        json_content = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if not json_content:
            logger.error(f"No JSON block found in response for batch {batch_num}")
            return []

        json_content = json_content[0].strip()
        try:
            generated_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for batch {batch_num}: {str(e)}")
            return []

        # Filter unique and valid messages
        for item in generated_data:
            message = item.get("original_message")
            if not message or message in generated_messages:
                continue
            if item.get("intent") not in ["Person", "Product", "Place"]:
                continue
            generated_messages.add(message)
            results_list.append({
                "original_message": message,
                "intent": item.get("intent"),
                "entities": json.dumps(item.get("entities", None))
            })

        logger.info(f"Generated {len(results_list)} unique messages in batch {batch_num}")
        return results_list

    except Exception as e:
        logger.error(f"Error generating messages for batch {batch_num}: {str(e)}")
        return []

def save_results(all_results, filename_prefix="generated_whatsapp_messages_vertex"):
    """
    Save the current results to a CSV file with a timestamp to avoid overwriting.
    """
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(filename, index=False)
        logger.info(f"Saved {len(all_results)} messages to {filename}")
        print(f"Saved {len(all_results)} messages to {filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        print(f"Failed to save CSV: {str(e)}")

def signal_handler(sig, frame, all_results):
    """
    Handle termination signals (e.g., SIGTERM) and save current results.
    """
    logger.info(f"Received termination signal: {sig}. Saving current results...")
    save_results(all_results)
    sys.exit(0)

def generate_whatsapp_messages(total_rows, batch_size, model):
    all_results = []
    all_messages = set()
    batch_num = 0

    # Register signal handlers for graceful termination
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, all_results))
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, all_results))

    try:
        while len(all_results) < total_rows:
            batch_num += 1
            remaining_rows = total_rows - len(all_results)
            current_batch_size = min(batch_size, remaining_rows)

            batch_results = generate_whatsapp_messages_batch(current_batch_size, batch_num, model)
            for result in batch_results:
                message = result["original_message"]
                if message not in all_messages:
                    all_messages.add(message)
                    all_results.append(result)

            logger.info(f"Total unique messages so far: {len(all_results)}")
            time.sleep(2)
            if len(all_results) % 1000 == 0 and all_results:
                save_results(all_results, f"temp_whatsapp_messages_{len(all_results)}")

        return all_results[:total_rows]

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Saving current results...")
        save_results(all_results)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during generation: {str(e)}")
        save_results(all_results)
        raise

if __name__ == "__main__":
    project_id = "sandbox-451305"
    location = "us-central1"
    model_name = "gemini-2.5-flash"
    total_rows = 700
    batch_size = 25
    all_results = []  # Initialize all_results to ensure it's defined even if authentication fails

    try:
        model = configure_vertex_ai(project_id, location, model_name, all_results)
    except SystemExit:
        exit(1)  # Exit if authentication or initialization fails
    except Exception as e:
        logger.error(f"Unexpected error during Vertex AI initialization: {str(e)}")
        print(f"Unexpected error during Vertex AI initialization: {str(e)}")
        exit(1)

    try:
        results_list = generate_whatsapp_messages(total_rows, batch_size, model)
        results_df = pd.DataFrame(results_list)
        print("\nGenerated Messages Preview:")
        print(results_df.head())
        save_results(results_list, "generated_whatsapp_messages_vertex")
    except Exception as e:
        logger.error(f"Failed to generate or save messages: {str(e)}")
        print(f"Failed to generate or save messages: {str(e)}")
        exit(1)