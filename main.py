import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import random
from datetime import datetime

app = FastAPI(title="NIDS Shield SOC")

# Mounting root for images
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    # FIX: Added encoding="utf-8" to handle special characters/emojis
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

# FIX: Added the missing logs endpoint for the live feed
@app.get("/api/logs")
async def get_logs():
    threats = ["TCP SYN Scan", "UDP Port Sweep", "ICMP Echo Request", "Benign Flow", "Service Discovery"]
    return [
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": random.choice(threats),
            "ip": f"192.168.1.{random.randint(1, 255)}",
            "risk": "HIGH" if "Scan" in threats[0] else "LOW"
        } for _ in range(5)
    ]

if __name__ == "__main__":
    print("🛡️ SOC Shield Interface starting at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)