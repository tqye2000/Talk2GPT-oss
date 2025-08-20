# PowerShell script for Talk2GPT-oss memory management
# Run this script if you encounter CUDA out of memory errors

Write-Host "Talk2GPT-oss Memory Management Script" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# Function to check if Python is available
function Test-Python {
    try {
        $null = python --version
        return $true
    }
    catch {
        return $false
    }
}

# Function to stop Streamlit processes
function Stop-StreamlitProcesses {
    Write-Host "Stopping any running Streamlit processes..." -ForegroundColor Yellow
    try {
        Get-Process | Where-Object {$_.ProcessName -like "*streamlit*" -or $_.CommandLine -like "*app.py*"} | Stop-Process -Force
        Write-Host "✅ Streamlit processes stopped" -ForegroundColor Green
    }
    catch {
        Write-Host "ℹ️ No Streamlit processes found" -ForegroundColor Blue
    }
}

# Function to clear Python cache
function Clear-PythonCache {
    Write-Host "Clearing Python cache..." -ForegroundColor Yellow
    
    # Clear __pycache__ directories
    Get-ChildItem -Path . -Name "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Clear .pyc files
    Get-ChildItem -Path . -Name "*.pyc" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
    
    Write-Host "✅ Python cache cleared" -ForegroundColor Green
}

# Function to run memory cleanup
function Invoke-MemoryCleanup {
    if (Test-Python) {
        Write-Host "Running memory cleanup script..." -ForegroundColor Yellow
        python memory_monitor.py --cleanup
    } else {
        Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    }
}

# Function to show memory status
function Show-MemoryStatus {
    if (Test-Python) {
        Write-Host "Checking memory status..." -ForegroundColor Yellow
        python memory_monitor.py --status
    } else {
        Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    }
}

# Main menu
do {
    Write-Host ""
    Write-Host "Choose an option:" -ForegroundColor Cyan
    Write-Host "1. Stop Streamlit processes"
    Write-Host "2. Clear Python cache"
    Write-Host "3. Run memory cleanup"
    Write-Host "4. Show memory status"
    Write-Host "5. Full cleanup (all of the above)"
    Write-Host "6. Exit"
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-6)"
    
    switch ($choice) {
        "1" { Stop-StreamlitProcesses }
        "2" { Clear-PythonCache }
        "3" { Invoke-MemoryCleanup }
        "4" { Show-MemoryStatus }
        "5" {
            Stop-StreamlitProcesses
            Clear-PythonCache
            Invoke-MemoryCleanup
            Show-MemoryStatus
            Write-Host ""
            Write-Host "✅ Full cleanup completed!" -ForegroundColor Green
        }
        "6" { 
            Write-Host "Exiting..." -ForegroundColor Yellow
            break 
        }
        default { Write-Host "Invalid choice. Please enter 1-6." -ForegroundColor Red }
    }
} while ($choice -ne "6")

Write-Host ""
Write-Host "Memory management script completed." -ForegroundColor Green
Write-Host "You can now restart the Talk2GPT-oss application." -ForegroundColor Green
