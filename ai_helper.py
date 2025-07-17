# ai_helper.py

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def generate_prediction_event(country: str, market: str, subcategory: str) -> list:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    url = "https://api.perplexity.ai/chat/completions"

    prompt = f"""
    请根据以下信息生成5个合理的未来事件预测主题：
    - 国家或地区：{country}
    - 市场大类：{market}
    - 子类：{subcategory}
    要求：
    - 每个事件要具体、可验证，具有时间范围（如“到2026年”）
    - 输出格式为 JSON 数组，例如：
      ["事件1", "事件2", "事件3", "事件4", "事件5"]
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "你是一个事件生成专家，只输出JSON数组"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return []
        else:
            print("API请求失败:", response.status_code)
            return []
    except Exception as e:
        print("请求异常:", str(e))
        return []