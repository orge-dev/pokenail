import random
import string
import datetime
import os
import pickle

def generate_timestamped_id():
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    
    # Generate random string of letters and digits
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    return timestamp + random_str

def analyze_checkpoints(checkpoint_dir="checkpoints"):
    """Read all agent state files and print visited coordinates stats."""
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("agent_state_") and filename.endswith(".pkl"):
            filepath = os.path.join(checkpoint_dir, filename)
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                visited_coords = state['visited_coords']
                print(f"{filename}: {len(visited_coords)} unique coordinates visited")

if __name__ == "__main__":
    analyze_checkpoints()
