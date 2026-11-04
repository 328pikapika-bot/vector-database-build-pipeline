@echo off
setlocal
cd /d "%~dp0"
python scripts\local_vector_monitor.py --host 127.0.0.1 --port 8788
