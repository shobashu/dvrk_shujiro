import serial
import time
import csv
import sys
import os

# --- CONFIGURATION ---
ARDUINO_PORT = '/dev/ttyACM0'  # Change this to your actual port (e.g., 'COM3' on Windows)
BAUD_RATE = 9600
DEFAULT_CSV_FILENAME = 'experiment_data.csv'

def write_header_if_needed(filename):
    """Creates the file and writes the header if it doesn't already exist."""
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'Target_Peg', 'Target_Color', 'Unix_Cue_Time', 'Unix_Lift_Time', 'Unix_Place_Time'])

def main():
    try:
        print(f"Connecting to {ARDUINO_PORT}...")
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) # Give Arduino time to reset
        
        current_csv_filename = DEFAULT_CSV_FILENAME
        write_header_if_needed(current_csv_filename)
        
        print("\n--- Python Monitor Ready ---")
        print(f"Default data file: {current_csv_filename}")
        print("Type 's' and press Enter to START the experiment.")
        print("Type 'c' and press Enter to CONTINUE after the break.")
        print("Type 'd' and press Enter for DEMO.")
        print("Type 't' and press Enter for TESTING.")
        print("Type 'q' and press Enter to ABORT.")
        print("----------------------------\n")

        python_sync_time = 0
        arduino_sync_millis = 0
        
        # Buffer for Windows input accumulation
        win_input_buffer = ""

        while True:
            # 1. Listen for keyboard input from the user (non-blocking)
            if sys.platform != "win32":
                import select
                i, o, e = select.select([sys.stdin], [], [], 0.0001)
                if i:
                    cmd = sys.stdin.readline()
                    if cmd:
                        arduino.write(cmd.encode('utf-8'))
            else:
                # Windows fallback for terminal input (non-blocking character accumulation)
                import msvcrt
                if msvcrt.kbhit():
                    char = msvcrt.getche()
                    if char in (b'\r', b'\n'):
                        print() # newline
                        arduino.write((win_input_buffer + '\n').encode('utf-8'))
                        win_input_buffer = ""
                    else:
                        try:
                            win_input_buffer += char.decode('utf-8')
                        except UnicodeDecodeError:
                            pass

            # 2. Listen for output from Arduino
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                if not line:
                    continue
                
                # Catch the dynamic filename tag from Arduino
                if line.startswith("FILENAME,"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        current_csv_filename = parts[1].strip()
                        write_header_if_needed(current_csv_filename)
                        print(f"[SYSTEM] Log file set to: {current_csv_filename}")
                
                # If Arduino sends a SYNC ping, record our current Unix time
                elif line.startswith("SYNC,"):
                    parts = line.split(",")
                    if len(parts) > 1:
                        arduino_sync_millis = int(parts[1])
                        python_sync_time = time.time()
                
                # If Arduino sends DATA, do the math and write to CSV
                elif line.startswith("DATA,"):
                    parts = line.split(",")
                    if len(parts) >= 7:
                        trial = parts[1]
                        peg = parts[2]
                        color = parts[3]
                        
                        # Convert Arduino millis to Laptop Unix Time
                        cue_unix = python_sync_time + ((int(parts[4]) - arduino_sync_millis) / 1000.0)
                        lift_unix = python_sync_time + ((int(parts[5]) - arduino_sync_millis) / 1000.0)
                        place_unix = python_sync_time + ((int(parts[6]) - arduino_sync_millis) / 1000.0)
                        
                        # Open, append, and close the file quickly
                        with open(current_csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([trial, peg, color, f"{cue_unix:.3f}", f"{lift_unix:.3f}", f"{place_unix:.3f}"])
                        
                        print(f"[CSV LOGGED] Trial {trial} data saved to {current_csv_filename}.")
                
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