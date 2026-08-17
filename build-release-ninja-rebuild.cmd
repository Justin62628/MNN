@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b %errorlevel%
cmake --build build-release-ninja --target tariff_mnn MNNConvert -j 4 > build-release-ninja-rebuild.log 2>&1
exit /b %errorlevel%
