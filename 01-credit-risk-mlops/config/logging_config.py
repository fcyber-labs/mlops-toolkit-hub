from loguru import logger
import sys
import os

# Remove default logger
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Add file handler
logger.add(
    "logs/credit_risk_mlops.log", rotation="10 MB", retention="1 week", level="DEBUG"
)
