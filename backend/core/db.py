from motor.motor_asyncio import AsyncIOMotorClient
from os import getenv
from dotenv import load_dotenv

load_dotenv()

client = AsyncIOMotorClient(getenv("MONGODB_URI"))
db = client["knowledge_chat"]
 
documents = db["documents"]
chunks = db["chunks"]
conversations = db["conversations"]
files = db["files"]  # original uploaded files, stored as binary
 
async def ping_database():
     await client.admin.command("ping")