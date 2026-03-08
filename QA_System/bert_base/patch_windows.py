import os
import platform
import signal

# If running on Windows platform, simple handling of SIGINT
if platform.system() == 'Windows':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
    signal.signal(signal.SIGINT, lambda x, y: None) 