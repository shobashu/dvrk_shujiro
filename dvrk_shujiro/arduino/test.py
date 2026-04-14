import tkinter as tk
import threading
import time

def trigger():
    time.sleep(2)
    print('Triggering popup...')
    root.after(0, show)

def show():
    win = tk.Toplevel(root)
    win.title('TEST POPUP')
    win.geometry('300x150+500+400')
    win.configure(bg='green')
    win.attributes('-topmost', True)
    win.lift()
    tk.Label(win, text='IT WORKS!', bg='green', fg='white').pack(pady=40)
    win.after(3000, win.destroy)

root = tk.Tk()
root.geometry('1x1+0+0')  # tiny, nearly invisible instead of hidden
root.configure(bg='black')
threading.Thread(target=trigger, daemon=True).start()

def keep_alive():
    root.after(100, keep_alive)
root.after(100, keep_alive)

print('Waiting 2 seconds...')
root.mainloop()