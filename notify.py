from plyer import notification

def send_alert(alert_message):
    notification.notify(
        title="🌩️ Disaster Alert Detected!",
        message=alert_message,
        timeout=5
    )
