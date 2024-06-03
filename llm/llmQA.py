import json
import logging
import argparse
import openai
from openai import OpenAI
import time
import socket
import os

from llms.remote import RemoteLLMs



def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", type=str, default="configs/gpt35.json")
    parser.add_argument("--prompt_path", type=str, default=r"E:\自然语言处理\QwLLM1\llms\prompt_zh_ryh.json")
    parser.add_argument("--question_path", type=str, default=r"E:\自然语言处理\QwLLM1\llms\question_zh_ryh.json")
    args = parser.parse_args()
    return args


class ChatGPTLLM(RemoteLLMs):
    def init_local_client(self):
        try:
            self.model = self.args['model']
            client = OpenAI(api_key=self.args['api_key'], base_url=self.args['base_url'])
            return client
        except:
            return None

    def create_prompt(self, prompt_path, context):
        if context is None:
            context = []
        #将prompt加入到system消息中
        with open(prompt_path, 'r',encoding="utf-8") as file:
            data = json.load(file)

            # 提取 prompt
        prompt = data['prompt']
        # print("Prompt:", prompt)
        context.append(
            {
                "role": "system",
                "content": prompt,
            }
        )
        return context

    def change_prompt(self, prompt_path, context):
        # 将prompt加入到system消息中
        with open(prompt_path, 'r',encoding="utf-8") as file:
            data = json.load(file)

        # 更改 prompt
        prompt = data['prompt']
        # print("Prompt:", prompt)
        # if context["role"] == "system":
        #     context["content"] = prompt
        for item in context:
            if item.get("role") == "system":
                item["content"] = prompt
                break  # 找到并更新后就退出循环
        return context



    def create_contexts(self, current_query, context=None):
        if context is None:
            context = []
        context.append(
            {
                "role": "user",
                "content": current_query,
            }
        )
        return context


    def create_batch_contexts(self, question,context):
        context.append({
                "role": "user",
                "content": question,
            })
        return context

    def request_llm(self, context, seed=1234, sleep_time=1, repeat_times=0):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=context,
                    stream=False,
                    seed=seed+repeat_times
                )
                context.append(
                    {
                        'role': response.choices[0].message.role,
                        'content': response.choices[0].message.content
                    }
                )
                return response
            except openai.RateLimitError as e:
                logging.error(str(e))
                raise e
            except (openai.APIError, openai.InternalServerError, socket.timeout) as e:
                logging.error(str(e))
                raise e
            except Exception as e:
                # 捕捉未预料的异常，考虑是否终止循环或做其他处理
                logging.error(f"An unexpected error occurred: {str(e)}")
                raise e
            time.sleep(sleep_time)

if __name__ == '__main__':
    # https://platform.openai.com/docs/api-reference
    dialogue_flag = True# 是否要继续对话标志 True:继续对话 False:退出对话
    Mode_flag = 1 # 1:多轮对话模式 2:批处理模式
    contexts = []# 对话上下文
    #获得初始的参数配置
    args = read_args()
    print(args)
    llm = ChatGPTLLM(args.llm_path)
    while True:
        user_mode = input("请输入你要选择的模式（输入“1”为对话模式，输入“2”为批处理模式,输入“END”退出。）:")
        if(user_mode=="1"):
            while True:
                dialogue_flag,contexts,args = llm.interactive_dialogue(args,contexts)
                if (dialogue_flag):
                    llm = ChatGPTLLM(args.llm_path)
                    print("已完成修改model")
                    continue
                else:
                    print("已结束当前对话，请重新输入模式。")
                    break
        elif(user_mode=="2"):
            llm.batch_dialogue(args)
        elif(user_mode=="END"):
            print("已退出对话系统。")
            break
        else:
            print("输入的模式错误，请重新输入！")
            continue