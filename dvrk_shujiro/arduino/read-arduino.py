import serial
import csv
import threading
from datetime import datetime, timedelta

# Configure your settings
PORT = '/dev/ttyACM1'  # Change to your Arduino port
BAUD = 9600
FILENAME = "arduino_data.csv"

# Initialize serial connection
ser = serial.Serial(PORT, BAUD, timeout=1)

initialized = False
startTime = datetime.now()
initialMilliSeconds = 0

def read_from_arduino():
    global initialized
    global initialMilliSeconds
    global startTime
    """Background task to read data and save to CSV."""
    with open(FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        print(f"--- Logging started. Saving to {FILENAME} ---")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    line = line.split(',')
                    if not initialized:
                        # the current reading should be synced with the current time.
                        startTime = datetime.now()
                        currTime = startTime
                        initialized = True
                        initialMilliSeconds = int(line[2])
                    else:
                        currTime = startTime + timedelta(milliseconds=(int(line[2]) - initialMilliSeconds))
                    line.append(currTime.timestamp())
                    # Print to your terminal so you can see it live
                    print(f"\n[Arduino]: {line}")
                    # Split comma-separated values and save
                    writer.writerow(line)
                    f.flush() # Ensure it writes to disk immediately

# Start the reading thread
thread = threading.Thread(target=read_from_arduino, daemon=True)
thread.start()

# Main loop for sending data
print("--- Type your command and press Enter to send to Arduino ---")
try:
    while True:
        cmd = input("> ") # This waits for your typing
        ser.write((cmd + '\n').encode('utf-8'))
except KeyboardInterrupt:
    print("\nClosing connection...")
    ser.close()
