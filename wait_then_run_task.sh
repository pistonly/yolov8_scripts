#!/bin/bash

# 指定要检查的进程ID
PID=2471

# 循环检查指定的进程是否存在
while true; do
    # 使用 ps 命令检查指定的 PID 是否存在
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "进程 $PID 已结束。"
        # 在这里放置你想要执行的命令
        echo "执行指定的命令..."
        # 例如，你可以在这里调用其他脚本或程序
        # ./your-command.sh
        ./train.sh
        break
    else
        echo "进程 $PID 仍在运行..."
    fi
    # 等待一段时间后再次检查，例如每5秒检查一次
    sleep 60
done
