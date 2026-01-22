#!/bin/bash

# 检查 browser_profile 目录是否已存在
if [ ! -d "$HOME/browser_profile" ]; then
    echo "browser_profile 不存在，正在拷贝 Chrome 配置..."
    mkdir -p ~/browser_profile
    cp -r ~/Library/Application\ Support/Google/Chrome/ ~/browser_profile
    echo "拷贝完成"
else
    echo "browser_profile 已存在，跳过拷贝步骤"
fi

# 启动 Chrome 调试模式
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome_profile"