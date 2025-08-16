@echo off
echo GPU Memory Cleanup Script for Talk2GPT-oss
echo ==========================================
echo.

echo Checking current memory status...
python memory_monitor.py --status
echo.

echo Performing emergency cleanup...
python memory_monitor.py --cleanup
echo.

echo Memory status after cleanup:
python memory_monitor.py --status
echo.

echo Cleanup completed! You can now restart the application.
pause
