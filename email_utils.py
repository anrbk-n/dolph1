from fastapi import BackgroundTasks, HTTPException
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()  

# SMTP settings --------------------------------------------------------------
SMTP_SERVER = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SMTP_USER")           
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")      

SUBJECT = "Email confirmation"                
# ---------------------------------------------------------------------------


def _send_email_sync(receiver: str, confirmation_link: str) -> None:
    """Send the confirmation e-mail synchronously (called in a background task)."""
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver
    msg["Subject"] = SUBJECT

    body = (
        "Здравствуйте!\n\n"
        "Пожалуйста, подтвердите свою почту, перейдя по ссылке:\n"
        f"{confirmation_link}\n\n"
        "Если вы не запрашивали регистрацию, просто проигнорируйте это письмо."
    )
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, receiver, msg.as_string())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send confirmation e-mail: {exc}",
        ) from exc


def send_confirmation_email(
    background_tasks: BackgroundTasks,
    user_email: str,
    confirmation_link: str,
) -> None:
    """
    Schedule asynchronous sending of the confirmation e-mail.

    background_tasks  – FastAPI BackgroundTasks instance  
    user_email        – recipient address  
    confirmation_link – link the user must click to confirm
    """
    background_tasks.add_task(_send_email_sync, user_email, confirmation_link)
