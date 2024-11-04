import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from rouge import Rouge
import jieba
from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
def original_mode():
    chat_model = ChatModel()
    messages = []
    print("欢迎使用 PediatricsGPT，输入 'clear' 清除历史记录，输入 'exit' 退出应用程序。")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("检测到输入时的解码错误，请设置终端编码为utf-8。")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("历史记录已清除。")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
        

    print("完成！")

    

if __name__ == "__main__":
    original_mode()
