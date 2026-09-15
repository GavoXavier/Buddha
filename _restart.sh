#!/bin/sh
# Restart the bot with the current source, appending to the same log.
# `>>` and `2>&1` matter: logging goes to stderr, and a plain redirect would
# both truncate the history and leave bot.log empty.
set -e
cd /c/Users/Admin/Documents/TradingView/POCKET

OLD=$(powershell -NoProfile -Command "(Get-Process python -ErrorAction SilentlyContinue | Select-Object -First 1).Id" | tr -d '\r')
if [ -n "$OLD" ]; then
  echo "stopping pid $OLD"
  powershell -NoProfile -Command "Stop-Process -Id $OLD -Force"
  sleep 2
fi

echo "=== starting new process ==="
powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Python314\python.exe -u main.py >> bot.log 2>&1' -WorkingDirectory 'C:\Users\Admin\Documents\TradingView\POCKET' -WindowStyle Hidden"
sleep 8
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,StartTime | Format-Table -AutoSize"
