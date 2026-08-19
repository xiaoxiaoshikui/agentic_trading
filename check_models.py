import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

def check_available_models():
    try:
        # 尝试从环境变量获取 key，如果没加载到则提示
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ 警告: 未找到 OPENAI_API_KEY 环境变量。尝试直接初始化...")
        
        client = OpenAI()
        print("正在连接 OpenAI API 获取可用模型列表...")
        
        # 获取所有模型
        models = client.models.list()
        
        # 筛选出主要对话模型 (gpt-3.5, gpt-4, o1 等)
        chat_models = [
            m.id for m in models.data 
            if ("gpt" in m.id or "o1" in m.id) and "instruct" not in m.id and "realtime" not in m.id and "audio" not in m.id
        ]
        chat_models.sort()
        
        print(f"\n✅ 您的 API Key 当前可用的主要模型:")
        print("=" * 50)
        
        # 分类显示
        categories = {
            "GPT-4 (旗舰)": lambda x: "gpt-4" in x and "preview" not in x,
            "GPT-4 (预览/Turbo)": lambda x: "gpt-4" in x and "preview" in x,
            "GPT-3.5 (经济)": lambda x: "gpt-3.5" in x,
            "O1 (推理系列)": lambda x: "o1" in x,
        }
        
        displayed = set()
        
        for category, condition in categories.items():
            subset = [m for m in chat_models if condition(m)]
            if subset:
                print(f"\n--- {category} ---")
                for model in subset:
                    print(f"  • {model}")
                    displayed.add(model)
        
        # 显示其他的
        others = [m for m in chat_models if m not in displayed]
        if others:
            print(f"\n--- 其他模型 ---")
            for model in others:
                print(f"  • {model}")
                
        print("=" * 50)
        print("\n推荐使用: gpt-4o")

    except Exception as e:
        print(f"\n❌ 获取模型列表失败: {e}")
        print("这可能意味着您的 API Key 无效，或者网络连接有问题。")

if __name__ == "__main__":
    check_available_models()
