import streamlit as st
import geocoder

# Initialize location variables
location = None

# Step 1: Add title and description
st.title("🚨 Emergency Alert System")
st.write("This system will track your location and send an alert to your emergency contact.")

# Step 2: Ask for emergency contact number
contact_number = st.text_input("Enter your emergency contact number:")

# Step 3: Capture the user's location
def get_location():
    # Geocoder will fetch your current location using your IP address
    g = geocoder.ip('me')
    return g.latlng  # Return latitude and longitude

if st.button("Get Location"):
    location = get_location()
    if location:
        st.write(f"📍 Your location: Latitude: {location[0]}, Longitude: {location[1]}")
    else:
        st.write("❌ Unable to fetch location.")

# Step 4: Display emergency contact and location details when ready
if location:
    st.write(f"Your emergency contact is: {contact_number}")
    st.write(f"Location: {location}")

# Step 5: Allow user to send alert
if st.button("Send Emergency Alert"):
    if contact_number and location:
        # Here, you can add code to send the emergency alert
        st.success("🚨 Emergency alert sent successfully!")
    else:
        st.warning("Please enter both contact number and location before sending an alert.")
