# 基于Prompt工程使用大模型完成特定任务

## 项目介绍：

基于本地部署大模型（Qwen-7B-Chat）和远程大模型（ChatGPT），结合给定的特定Prompt，完成特定的任务（项目中任务为给定一段背景材料和一个问题， 要求模型写出对应的答案以及理由。）。

* 支持多轮对话
* 支持对话时切换LLM，prompt
* 支持批量读取批量输出（支持断点续传）

## 环境要求：

* python 3.8及以上版本

* pytorch 1.12及以上版本，推荐2.0及以上版本

* transformers 4.32及以上版本

* 建议使用CUDA 11.4及以上（GPU用户需考虑此选项）

具体步骤如下：

### 1.创建conda虚拟环境

  ```
conda create -n your_env_name python=3.xx
  ```

### 2.激活虚拟环境

  ```
conda activate your_env_name
  ```

### 3.下载依赖

  ```
pip install -r requirements.txt
  ```

注：requirements.txt中已经包含modelscope库

## 本地模型部署：

### 运行python下载模型到本地

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-7B-Chat',revision ='模型版本',cache_dir='模型保存路径')

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

history=None

while True:
    message = input('User:')
    response, history = model.chat(tokenizer, message, history=history)
    print('System:',end='')
    print(response)
```

上面代码运行后，等待模型下载。模型下载完成后如果可以正常对话则部署成功。

例如：

```
User:你好
你好
System:你好！有什么我能为你做的吗？
```

## 修改路径/配置

在运行代码前，需要修改代码中的路径/配置。

修改远程模型的配置，具体为`llms/remote/configs/gpt35.json`中的`model`，`api_key`和`base_url`。

修改`openai_api.py`中的模型路径改为本地部署模型的路径。

## 运行

先启动本地模型的api服务，运行`openai_api.py`：

```
python openai_api.py
```

然后运行对话系统`llmQA.py`：

```
python llmQA.py  --llm_path configs/gpt35.json --prompt_path you_prompt_path --question_path you_question_path
```

