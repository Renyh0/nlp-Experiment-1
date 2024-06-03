import json
import re

class RemoteLLMs:

    def init_local_client(self):
        """
        初始化用户的客户端
        :param args:
        :return:
        """
        print("RemoteLLMs")
        raise NotImplementedError()

    def __load_args(self, config_path):
        # 首先读取Config
        self.args = json.load(open(config_path))

    def __init__(self, config_path):
        # 首先读取Config
        self.__load_args(config_path)
        self.max_retries = self.args.get("max_retries", 5)
        self.client = None
        for idx in range(self.max_retries):
            model = self.init_local_client()
            if model is not None:
                self.client = model
                break
        if self.client is None:
            raise ModuleNotFoundError()

    def create_contexts(self, history, current_query):
        pass

    def create_prompt(self, prompt_path, context):
        pass


    def change_prompt(self, prompt_path, context):
        pass


    # def create_batch_prompt(self, history, current_query):
    #     pass

    def create_batch_contexts(self, question ,context):
        pass


    def request_llm(self, context, seed=1234, sleep_time=1, repeat_times=0):
        pass

    def fit_case(self, pattern: str, data: dict, meta_dict: dict = None):

        if meta_dict is not None:
            for k,v in meta_dict.items():
                pattern = pattern.replace(k, str(v))

        for k, v in data.items():
            pattern = pattern.replace(k, str(v))

        assert '{{' not in pattern, pattern
        assert '}}' not in pattern, pattern
        return pattern

    def extract_answers_and_reasons(self,text):
        pattern = r"答案：(.*?)\n理由：(.*?)$"
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        extracted_text = ""
        for match in matches:
            answer = match[0].strip()
            reason = match[1].strip()
            extracted_text += f"答案：{answer}\n理由：{reason}\n\n"
        return extracted_text.strip()


    def interactive_dialogue(self,args,contexts):
        """
        进行交互式的对话，进行模型检查
        :return:
        """

        # 如果是首次进入对话，首先获取prompt
        if(contexts==[]):
            contexts = self.create_prompt(args.prompt_path,contexts)

        while True:
            current_query = input("请输入当前你的对话(输入'CLEAN'清除上下文，'END'离开对话，‘CHANGE_ARGS’修改prompt或model)：")
            if current_query == "CLEAN":
                contexts = []
                print("已经清除上下文")
                continue
            elif current_query == "END":
                return False,None,None
            elif current_query == "CHANGE_ARGS":
                current_args = input("请选择你要修改的参数(输入'prompt'修改prompt，'model'修改模型)：")
                if current_args == "prompt":
                    current_args = input("请输入prompt参数配置文件路径")
                    if(args.prompt_path != current_args):
                        args.prompt_path = current_args
                        contexts = self.change_prompt(args.prompt_path, contexts)
                        print("已完成修改prompt")
                        continue
                    else:
                        print("和当前使用的prompt文件一致")
                        continue
                elif current_args == "model":
                    current_args = input("请输入model参数配置文件路径")
                    if (args.llm_path != current_args):
                        args.llm_path = current_args
                        return True,contexts,args
                    else:
                        print("和当前使用的model文件一致")
                        continue

            contexts = self.create_contexts(current_query, contexts)
            print("contexts")
            print(contexts)
            results = self.request_llm(contexts)
            # print("%s\t%s" % (results[-1]['role'], results[-1]['content']))
            print(results.choices[0].message.content)




    def batch_dialogue(self,args):
        """
        批量处理
        :param prompt_path:输入的prompt文件的路径,question_path:输入的问题文件的路径
        :return:
        """
        contexts = []
        result_data = []# 存储结果的列表

        if (contexts == []):
            contexts = self.create_prompt(args.prompt_path, contexts)


        # 读取 JSON 文件
        with open(args.question_path, 'r',encoding='utf-8') as file:
            data = json.load(file)

        # 提取并存放 egx 部分
        question = []
        for key, value in data['question'].items():
            if key.startswith('eg'):
                question.append(value)


        try:
            with open(r"E:\自然语言处理\QwLLM1\llms\batch_output.json", 'r', encoding='utf-8') as json_file:
                exist_data = json.load(json_file)
        except:
            exist_data=None
        exist_index = 0
        if(exist_data is not None) :
            exist_index = len(exist_data)
        if(exist_index<len(question)):
            print("前{}条问题已处理，将从第{}条问题开始处理".format(exist_index,exist_index+1))
        elif(exist_index==len(question)):
            print("所有问题已处理")


        # print("result_data:",result_data)
        # print("exist_index:",exist_index)
        # 输出 prompt 和 egx 的值
        for i, value in enumerate(question[exist_index:]):
            # print("第{}个问题: {}".format(i+1+exist_index, value))
            contexts = self.create_batch_contexts(question=value ,context=contexts)#加入question
            # print("contexts:")
            # print(contexts)
            results= self.request_llm(contexts)

            extracted_results = self.extract_answers_and_reasons(results.choices[0].message.content)

            print("第{}个问题: {}".format(i+1+exist_index, value))
            # print("原始回答: {}".format(results.choices[0].message.content))
            print("回答: {}".format(extracted_results))
            # 构建对话结构
            conversation_pair = {"conversations":
            [
            {
                "role": "user",
                "content": value
            },
            {
                "role": "assistant",
                "content": extracted_results
            }]}

            try:
                with open(r"E:\自然语言处理\QwLLM1\llms\batch_output.json", 'r', encoding='utf-8') as json_file:
                    result_data = json.load(json_file)
            except:
                result_data = []

            result_data.append(conversation_pair)

            #每次重置contexts
            contexts = []
            contexts = self.create_prompt(args.prompt_path, contexts)
            with open(r"E:\自然语言处理\QwLLM1\llms\batch_output.json", 'w' ,encoding='utf-8') as json_file:
                json.dump(result_data, json_file, indent=4,ensure_ascii=False)
            print("完成第{}个问题，数据已保存".format(i+1+exist_index))
        print("完成全部问题,结果保存到batch_output.json文件中")




