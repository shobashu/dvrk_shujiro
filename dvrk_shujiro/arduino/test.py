import tkinter as tk
import threading
import time
import signal
import sys

def trigger():
    time.sleep(2)
    print('Triggering popup...')
    root.after(0, show)

def show():
    win = tk.Toplevel(root)
    win.title('TEST POPUP')
    
    # Position on Monitor 0 (HDMI-1)
    monitor_x = 1280
    monitor_width = 3840
    monitor_height = 2160
    popup_width = 300
    popup_height = 150
    
    x = monitor_x + (monitor_width - popup_width) // 2
    y = (monitor_height - popup_height) // 2
    
    win.geometry(f'{popup_width}x{popup_height}+{x}+{y}')
    win.configure(bg='green')
    win.attributes('-topmost', True)
    win.lift()
    tk.Label(win, text='IT WORKS!', bg='green', fg='white', font=('Arial', 20)).pack(pady=40)
    win.after(3000, win.destroy)

def signal_handler(sig, frame):
    print('\nCtrl+C detected, exiting...')
    root.quit()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

root = tk.Tk()
root.geometry('1x1+0+0')
root.configure(bg='black')
threading.Thread(target=trigger, daemon=True).start()

def keep_alive():
    root.after(100, keep_alive)
root.after(100, keep_alive)

print('Waiting 2 seconds... (Press Ctrl+C to quit)')
root.mainloop()