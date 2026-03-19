# dVRK Data Collection Tools

**Author:** Shujiro  
**Project:** dVRK surgical robot data collection and analysis

## Description

Collection of ROS 2 tools for measuring task performance metrics on the da Vinci Research Kit (dVRK):

- **Task timer with GUI** - Visual progress bar showing active manipulation time
- **Path length calculator** (coming soon)
- **CSV data logger** (coming soon)

## Scripts

### `task_timer_gui`
Floating semi-transparent timer window that tracks time and total path length when MONO pedal is pressed.

**Features:**
- Dual-window display for stereo endoscope viewing
- Color-coded progress bar (green → yellow → red)
- Only counts active manipulation time (MONO pedal pressed)
- Configurable time limit (default: 2 minutes)

**Usage:**
```bash
ros2 run dvrk_shujiro task_timer_gui

