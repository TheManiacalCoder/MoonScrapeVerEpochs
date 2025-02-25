@echo off
echo Checking for changes...

git status

echo Adding all changes...
git add .

echo Committing changes...
set /p commit_message="Enter commit message: "
git commit -m "%commit_message%"

echo Setting remote repository...
git remote set-url origin https://github.com/TheManiacalCoder/MoonScrapeVerEpochs.git

echo Pushing to remote repository...
git push origin main

if %errorlevel% equ 0 (
    echo Push successful!
) else (
    echo Push failed. Check for errors.
)

pause 