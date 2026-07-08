import requests
import random
import time

API_URL = "http://127.0.0.1:8080/api"
MODALITIES = ["VISION", "MOTOR", "MEMORY", "DANGER"]

print("🦁 LionAI Simulation Environment Started")
print(f"Connecting to {API_URL}...")

tick = 0
while True:
    # Generate random sensory data
    inputs = []
    for mod in MODALITIES:
        if random.random() > 0.3:  # 70% chance to include a modality
            values = [random.uniform(-1.0, 1.0) for _ in range(64)]
            inputs.append({"modality": mod, "values": values})
    
    # Random previous reward
    prev_reward = random.uniform(-1.0, 1.0) if random.random() > 0.5 else 0.0

    payload = {
        "inputs": inputs,
        "prev_reward": prev_reward
    }
    
    try:
        response = requests.post(f"{API_URL}/tick", json=payload)
        data = response.json()
        print(f"Tick {data['tick']} | Action: {data['action']} | Stress: {data['stress']:.2f}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to server: {e}")
        time.sleep(2)
        continue
        
    tick += 1
    
    # Trigger sleep cycle every 50 ticks
    if tick % 50 == 0:
        print("\n🌙 Triggering Sleep Cycle...")
        try:
            sleep_res = requests.post(f"{API_URL}/sleep", json={"final_reward": 0.0})
            sleep_data = sleep_res.json()
            print(f"Evolution Occurred: {sleep_data['evolution_occurred']} | Gen: {sleep_data['generation']}\n")
        except requests.exceptions.RequestException as e:
            print(f"Failed to trigger sleep: {e}")
            
    time.sleep(0.5)  # 2 ticks per second
