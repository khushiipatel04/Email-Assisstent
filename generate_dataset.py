import pandas as pd
import random
import time

# --- Configuration ---
NUM_ENTRIES = 1000
OUTPUT_FILENAME = "emails.csv"

# --- Intent Categories and Templates ---
intents = {
    "FILE_REQUEST": [
        "Can you send me the {file_type}?",
        "Please forward the {file_type} when you get a chance.",
        "Following up on our chat, could you share the {file_type}?",
        "Where can I find the latest {file_type}?",
        "Need the {file_type} for the meeting.",
        "Requesting access to the {file_type}.",
        "Hi, is the {file_type} ready?",
        "Reminder to send the {file_type}.",
    ],
    "MEETING_REQUEST": [
        "Are you free to chat {time_frame}?",
        "Can we schedule a call to discuss {topic}?",
        "Let's sync up {time_frame}.",
        "Meeting request: {topic} discussion.",
        "Available for a quick call {time_frame}?",
        "Setting up a meeting for {topic}.",
        "How about we meet {time_frame}?",
        "Do you have time for a brief meeting?",
    ],
    "REFERRAL": [
        "You have a new job referral for the {role} position.",
        "Referral received from {person_name}.",
        "Update on your application: referred by {person_name}.",
        "Good news! {person_name} referred you for the {role} role.",
        "Please review the attached referral profile.",
        "ACTION REQUIRED: New referral submission.",
    ],
    "PROMOTION": [
        "Flash Sale! {discount}% off everything!",
        "Don't miss our {event} deals!",
        "Limited time offer: {product_name} now available.",
        "Weekly Digest: Top news and special offers.",
        "Exclusive {discount}% discount for you!",
        "Check out our new {product_name} lineup.",
        "Biggest Sale Ever! Up to {discount}% off.",
        "Your weekly newsletter and promotions.",
    ],
    "ACKNOWLEDGEMENT": [
        "Got it, thanks!",
        "Thanks for the update.",
        "Sounds good.",
        "Received, thank you.",
        "Okay, will do.",
        "Perfect, thanks!",
        "Thanks for sending that over.",
        "Acknowledged.",
    ],
    "SPAM": [
        "URGENT: Your account is locked! Click here!",
        "Congratulations! You won {amount}!",
        "Claim your free {prize} now!",
        "Make money fast online!",
        "Verify your bank details immediately!",
        "Viagra cheap online!",
        "Exclusive offer just for you - Limited spots!",
        "Your inheritance is waiting claim now!",
    ]
}

# --- Placeholder Data ---
file_types = ["report", "slides", "document", "spreadsheet", "presentation", "proposal", "agenda", "minutes"]
time_frames = ["tomorrow", "later today", "sometime this week", "next Monday", "at 3 PM", "around noon"]
topics = ["the project", "the Q3 results", "next steps", "the budget", "the new design", "our strategy"]
roles = ["Software Engineer", "Product Manager", "Data Analyst", "Marketing Lead", "Designer"]
person_names = ["Alex", "Ben", "Chloe", "David", "Emma", "Frank", "Grace", "Henry"] # Simple names for variety
discounts = ["10", "20", "30", "50", "70"]
events = ["Summer", "Holiday", "Weekend", "Clearance"]
product_names = ["gadget", "software", "service", "subscription", "widget"]
amounts = ["$1,000,000", "£5000", "a free iPhone"]
prizes = ["iPhone", "vacation", "gift card", "iPad"]

# --- Generation Logic ---
generated_data = []

# Simple subjects based on intent
def get_subject(intent):
    if intent == "FILE_REQUEST": return random.choice(["File Request", "Document Needed", "Regarding File"])
    if intent == "MEETING_REQUEST": return random.choice(["Meeting Request", "Quick Chat?", "Availability"])
    if intent == "REFERRAL": return random.choice(["New Referral", "Application Update", "Referral Submission"])
    if intent == "PROMOTION": return random.choice(["Special Offer!", "Don't Miss Out!", "Weekly Deals"])
    if intent == "ACKNOWLEDGEMENT": return random.choice(["Re: Update", "Got it", "Thanks"])
    if intent == "SPAM": return random.choice(["URGENT!", "You Won!", "Verify Now!", "Exclusive Offer"])
    return "Email Subject" # Default

for i in range(NUM_ENTRIES):
    intent_key = random.choice(list(intents.keys()))
    template = random.choice(intents[intent_key])

    # Fill placeholders - basic implementation
    body = template.format(
        file_type=random.choice(file_types),
        time_frame=random.choice(time_frames),
        topic=random.choice(topics),
        role=random.choice(roles),
        person_name=random.choice(person_names),
        discount=random.choice(discounts),
        event=random.choice(events),
        product_name=random.choice(product_names),
        amount=random.choice(amounts),
        prize=random.choice(prizes)
    )

    subject = get_subject(intent_key)

    generated_data.append({
        "subject": subject,
        "body": body,
        "intent": intent_key
    })

# --- Create DataFrame and Save ---
df_generated = pd.DataFrame(generated_data)

# Optional: Shuffle the data
df_generated = df_generated.sample(frac=1).reset_index(drop=True)

try:
    df_generated.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Successfully generated {NUM_ENTRIES} emails and saved to '{OUTPUT_FILENAME}'")
except Exception as e:
    print(f"Error saving file: {e}")
