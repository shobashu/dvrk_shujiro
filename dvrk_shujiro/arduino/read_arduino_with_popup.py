import serial
import time
import csv
import sys
import threading
import tkinter as tk

# ── Popup import ──────────────────────────────────────────────────────────────
sys.path.insert(0, '/home/stanford/dvrk_shujiro_ws/src/dvrk_shujiro')
from dvrk_shujiro.gui.trial_popup import TrialPopup

# --- CONFIGURATION ---
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE    = 9600
CSV_FILENAME = 'experiment_data.csv'


def main(popup):
    """
    Runs in a background thread.
    with popup.show_threadsafe() and popup.complete_threadsafe() added.
    """
    try:
        print(f"Connecting to {ARDUINO_PORT}...")
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)

        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'Target_Peg', 'Target_Color',
                             'Unix_Cue_Time', 'Unix_Lift_Time', 'Unix_Place_Time'])

            print("\n--- Python Monitor Ready ---")
            print(f"Data will be saved to: {CSV_FILENAME}")
            print("Type 's' and press Enter to START the experiment.")
            print("Type 'q' and press Enter to ABORT.")
            print("----------------------------\n")

            python_sync_time    = 0
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
                        print()

                # 2. Listen for output from Arduino
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    if not line:
                        continue

                    # If Arduino sends a SYNC ping, record our current Unix time
                    if line.startswith("SYNC,"):
                        parts = line.split(",")
                        arduino_sync_millis = int(parts[1])
                        python_sync_time    = time.time()

                    # If Arduino sends LIFTED, show the popup
                    elif line == "LIFTED":
                        print("[ARDUINO] Cylinder lifted!")
                        popup.show_threadsafe()

                    # If Arduino sends DATA, do the math and write to CSV
                    elif line.startswith("DATA,"):
                        parts      = line.split(",")
                        trial      = parts[1]
                        peg        = parts[2]
                        color      = parts[3]

                        # Convert Arduino millis to Laptop Unix Time
                        cue_unix   = python_sync_time + ((int(parts[4]) - arduino_sync_millis) / 1000.0)
                        lift_unix  = python_sync_time + ((int(parts[5]) - arduino_sync_millis) / 1000.0)
                        place_unix = python_sync_time + ((int(parts[6]) - arduino_sync_millis) / 1000.0)

                        # Write to CSV
                        writer.writerow([trial, peg, color,
                                         f"{cue_unix:.3f}",
                                         f"{lift_unix:.3f}",
                                         f"{place_unix:.3f}"])
                        file.flush()
                        print(f"[CSV LOGGED] Trial {trial} data saved.")
                        popup.complete_threadsafe()

                    else:
                        print(f"[ARDUINO] {line}")

    except serial.SerialException as e:
        print(f"Error connecting to {ARDUINO_PORT}: {e}")
    except Exception as e:
        print(f"[Arduino thread error] {e}")


# def main():
#     # ── Setup tkinter root (hidden) and popup ─────────────────────────────────
#     root = tk.Tk()
#     root.withdraw()
#     popup = TrialPopup(root=root)

#     # ── Start Arduino loop in background thread ───────────────────────────────
#     thread = threading.Thread(
#         target=arduino_loop,
#         args=(popup,),
#         daemon=True
#     )
#     thread.start()

#     # ── Keep tkinter alive with periodic update ───────────────────────────────
#     def keep_alive():
#         root.after(100, keep_alive)

#     root.after(100, keep_alive)

#     try:
#         root.mainloop()
#     except KeyboardInterrupt:
#         print("\nExiting...")
#     finally:
#         popup.print_final_score()


if __name__ == '__main__':
    main()