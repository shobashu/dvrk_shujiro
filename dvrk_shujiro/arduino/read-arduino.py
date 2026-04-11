import serial
import time
import csv
import sys

# --- CONFIGURATION ---
ARDUINO_PORT = '/dev/ttyACM0'  # Change this to your actual port (e.g., '/dev/ttyACM0' on Mac/Linux)
BAUD_RATE = 9600
CSV_FILENAME = 'experiment_data.csv'

def main():
    try:
        print(f"Connecting to {ARDUINO_PORT}...")
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) # Give Arduino time to reset
        
        # Open CSV file and write headers if it's a new file
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['Trial', 'Target_Peg', 'Target_Color', 'Unix_Cue_Time', 'Unix_Lift_Time', 'Unix_Place_Time'])
            
            print("\n--- Python Monitor Ready ---")
            print(f"Data will be saved to: {CSV_FILENAME}")
            print("Type 's' and press Enter to START the experiment.")
            print("Type 'q' and press Enter to ABORT.")
            print("----------------------------\n")

            python_sync_time = 0
            arduino_sync_millis = 0
            
            while True:
                # 1. Listen for keyboard input from the user (non-blocking)
                import select
                if sys.platform != "win32":
                    i, o, e = select.select([sys.stdin], [], [], 0.0001)
                    if i:
                        cmd = sys.stdin.readline().strip()
                        if cmd:
                            arduino.write(cmd.encode('utf-8'))
                else:
                    # Basic Windows fallback for terminal input (blocking)
                    import msvcrt
                    if msvcrt.kbhit():
                        cmd = msvcrt.getche().decode('utf-8')
                        arduino.write(cmd.encode('utf-8'))
                        print() # newline

                # 2. Listen for output from Arduino
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    if not line:
                        continue
                        
                    # If Arduino sends a SYNC ping, record our current Unix time
                    if line.startswith("SYNC,"):
                        parts = line.split(",")
                        arduino_sync_millis = int(parts[1])
                        python_sync_time = time.time()
                        
                    # If Arduino sends DATA, do the math and write to CSV
                    elif line.startswith("DATA,"):
                        parts = line.split(",")
                        trial = parts[1]
                        peg = parts[2]
                        color = parts[3]
                        
                        # Convert Arduino millis to Laptop Unix Time
                        cue_unix = python_sync_time + ((int(parts[4]) - arduino_sync_millis) / 1000.0)
                        lift_unix = python_sync_time + ((int(parts[5]) - arduino_sync_millis) / 1000.0)
                        place_unix = python_sync_time + ((int(parts[6]) - arduino_sync_millis) / 1000.0)
                        
                        # Write to CSV
                        writer.writerow([trial, peg, color, f"{cue_unix:.3f}", f"{lift_unix:.3f}", f"{place_unix:.3f}"])
                        file.flush() # Force save to disk immediately
                        print(f"[CSV LOGGED] Trial {trial} data saved.")
                        
                    else:
                        # Print all standard Arduino serial output to the console
                        print(f"[ARDUINO] {line}")

    except serial.SerialException as e:
        print(f"Error connecting to {ARDUINO_PORT}: {e}")
    except KeyboardInterrupt:
        print("\nExiting Python script...")
    finally:
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()

if __name__ == '__main__':
    main()
