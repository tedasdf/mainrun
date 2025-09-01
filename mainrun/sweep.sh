@echo off
setlocal enabledelayedexpansion

for /f "tokens=*" %%i in ('wandb sweep --project gpt-from-scratch --entity arc_agi sweep.yaml ^| findstr "arc_agi/gpt-from-scratch"') do (
    set SWEEP_ID=%%i
)

echo Sweep ID is !SWEEP_ID!

wandb agent !SWEEP_ID!
