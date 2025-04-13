import re
from datetime import datetime, timedelta
from functools import wraps
from cryptography.fernet import Fernet
import os

# Generate or load encryption key
if not os.path.exists("secret.key"):
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
else:
    with open("secret.key", "rb") as key_file:
        key = key_file.read()

cipher_suite = Fernet(key)

# Validate stock ticker symbols
def validate_ticker(ticker):
    return bool(re.match(r"^[A-Z]{1,5}$", ticker))

# Sanitize user input
def sanitize_input(input_str):
    return re.sub(r"[^a-zA-Z0-9]", "", input_str)

# Encrypt data
def encrypt_data(data):
    return cipher_suite.encrypt(data.encode()).decode()

# Decrypt data
def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data.encode()).decode()

# Rate limiting decorator
def rate_limit(max_calls, time_window):
    def decorator(func):
        calls = []
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            calls[:] = [call for call in calls if call > now - time_window]
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded. Please try again later.")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator