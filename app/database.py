from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

# Connect to MongoDB using the URL from your .env file
client = AsyncIOMotorClient(settings.mongodb_url)

# This is your database (MongoDB will auto-create it on first use!)
db = client["video_assistant_db"]

# This is your "users" collection (like a table in SQL)
users_collection = db["users"]
