from motor.motor_asyncio import AsyncIOMotorClient
from os import getenv
from dotenv import load_dotenv

load_dotenv()

client = AsyncIOMotorClient(getenv("MONGODB_URI"))
db = client["knowledge_chat"]
 
documents = db["documents"]
chunks = db["chunks"]
conversations = db["conversations"]
 
async def ping_database():
     await client.admin.command("ping")