import smtplib
from email.mime.text import MIMEText

def send_email(body, subject="AlphaZero Training Update", receiver_email="kolinhuang0428@gmail.com"):
    # Email details
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server (e.g., smtp.gmail.com for Gmail)
    smtp_port = 587  # Common port for TLS
    sender_email = "kolinhuang0428@gmail.com"
    sender_password = "puze wlyz poxf hzei"  # Use an app-specific password if required

    # Create the email
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade the connection to secure
            server.login(sender_email, sender_password)  # Log in to the SMTP server
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
    except Exception as e:
        print(f"Failed to send email: {e}")
