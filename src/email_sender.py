
import smtplib
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmailSender:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")

    def send_email(self, to_email: str, subject: str, body: str):
        logging.info(f"Attempting to send email to {to_email} with subject: '{subject}'")
        
        if not all([self.smtp_server, self.smtp_port, self.smtp_user, self.smtp_password]):
            logging.error("SMTP settings are missing. Please check your .env file.")
            return

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                message = f"Subject: {subject}\n\n{body}"
                server.sendmail(self.smtp_user, to_email, message)
            logging.info(f"Email successfully sent to {to_email}")
        except smtplib.SMTPAuthenticationError as e:
            logging.error(f"SMTP Authentication Error: {e}. Check your SMTP_USER and SMTP_PASSWORD.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while sending email: {e}")


