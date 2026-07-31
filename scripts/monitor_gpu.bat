@echo off
powershell -NoProfile -Command "while ($true) { Clear-Host; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv; Start-Sleep -Seconds 3 }"
