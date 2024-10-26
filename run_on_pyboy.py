

from pyboy import PyBoy
from config import ROM_PATH  # Import ROM_PATH from config


# Create a PyBoy instance with the ROM
pyboy = PyBoy(ROM_PATH)

try:
    # Main loop to run the emulator
    while True:
        # Update the emulator state
        if not pyboy.tick():
            break  # Exit if the emulator signals to stop

except KeyboardInterrupt:
    print("Program interrupted. Stopping emulator...")
finally:
    # Clean up and close the emulator
    pyboy.stop()
