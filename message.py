from twilio.rest import Client

# Twilio credentials from your account
TWILIO_ACCOUNT_SID = 'AC536b0705700c3b8498b8611008a285b7'
TWILIO_AUTH_TOKEN = '8af72aa07e7386a755add5fca2b92ba1'
TWILIO_PHONE_NUMBER = '9972316354'

# Function to send an SMS alert to the user
def send_sms_alert(user_phone_number, message):
    try:
        # Initialize the Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        user_phone_number=7970787292
        # Send the SMS
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,  # Your Twilio phone number
            to=user_phone_number       # The user's phone number
        )
        st.success(f"Alert sent successfully to {user_phone_number}!")
    except Exception as e:
        st.error(f"Error sending SMS: {e}")
