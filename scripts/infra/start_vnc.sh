#!/bin/bash
# 웹 VNC 서버 시작 스크립트

# Xvfb 시작 (가상 디스플레이)
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# 잠시 대기
sleep 2

# XFCE 데스크톱 환경 시작
startxfce4 &

# VNC 서버 시작 (비밀번호 없음)
x11vnc -display :99 -nopw -listen localhost -xkb -rfbport 5900 -forever -shared &

# 웹 VNC 서버 시작 (포트 6080)
websockify --web /usr/share/novnc/ 6080 localhost:5900 &

echo "=========================================="
echo "웹 VNC 서버가 시작되었습니다!"
echo "브라우저에서 다음 주소로 접속하세요:"
echo "http://localhost:6080/vnc.html"
echo "=========================================="

# AI2-THOR 실행
exec "$@"
