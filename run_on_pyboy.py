

from pyboy import PyBoy

# Path to your Game Boy ROM file
rom_path = "red.gb"  # Replace with the path to your ROM file

# Create a PyBoy instance with the ROM
pyboy = PyBoy(rom_path)

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
