# prompts.py

def classify_email(email_content):
    return f"""You are an AI assistant for GlobalMart's customer support team. Your task is to classify the following email into one of these categories: Order Issues, Product Inquiries, Technical Support, or Returns/Refunds.

Email content: {email_content}

Provide your classification as a single word or phrase, choosing from the categories mentioned above. Do not include any explanation or additional text.

Classification:"""

# You can add more prompt functions here in the future
