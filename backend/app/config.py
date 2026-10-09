import os

APP_NAME = os.environ.get("APP_NAME", "hackathon")
DEBUG = os.environ.get("DEBUG", "0") == "1"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
