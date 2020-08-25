# minGPT-group-bot-server
q q 群 暴 论 b o t

简单几步训练你群专属的暴论人工智能(zhang)！

![太智能了](https://raw.githubusercontent.com/DEDZTBH/minGPT-group-bot-server/master/demo.jpg)

步骤：
1. 从QQ导出txt格式的聊天记录，重命名为qbqljl.txt
2. 将其放在项目根目录，运行python data_preprocess.py并生成input.txt（语料库）。第一行有可能删不掉，如果显示日期时间则手动删除。
3. 将mingpt和input.txt打包为qbqljl.zip，然后上传到kaggle为Dataset。同时创建一个新notebook然后上传train_model.ipynb使用这个Dataset，开启gpu后保存运行，就可以白嫖kaggle的GPU训练啦！当然如果自己有设备也可以自己训练。
4. 下载model.pkl，放在项目根目录，之后运行python -m pip -r requirements.txt安装依赖。
5. 运行python predict_server.py 8000 启动后台服务器（8000为端口，如果需要可以换一个）
6. 前往https://github.com/DEDZTBH/minGPT-group-bot-mirai 到release页面下载mirai端。根据教程启动。
7. 你的bot可以使用了！

*由于license原因mirai端是一个单独的repo

*人工智能模型基于minGPT (https://github.com/karpathy/minGPT) 并使用GPT-1参数