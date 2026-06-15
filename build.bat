@echo off
echo =========================================
echo Budowanie obrazu Docker: tomato-ai...
echo =========================================
docker build -t tomato-ai .
echo.
echo Zbudowano pomyslnie!
pause