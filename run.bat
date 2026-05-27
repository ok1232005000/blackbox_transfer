@echo off
echo ================================
echo 对抗样本攻击系统启动器
echo ================================
echo.
echo 选择启动模式:
echo [1] 仅启动API服务
echo [2] 仅启动前端
echo [3] 同时启动API和前端
echo [4] 运行测试
echo.
set /p choice=请输入选择 (1-4):

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto start_frontend
if "%choice%"=="3" goto start_both
if "%choice%"=="4" goto run_tests
goto end

:start_api
echo 正在启动API服务...
python start_api.py
goto end

:start_frontend
echo 正在启动前端服务...
python start_frontend.py
goto end

:start_both
echo 正在启动API服务（后台）...
start /b python start_api.py
echo 正在启动前端服务...
python start_frontend.py
goto end

:run_tests
echo 正在运行测试...
python -m pytest tests/ -v
goto end

:end
pause
