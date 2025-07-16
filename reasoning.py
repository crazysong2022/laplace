# reasoning.py
import os
import re
import requests
from typing import Optional

class ReasoningService:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai/chat/completions"

    def generate_reasoning(self, event: str, predictions: list) -> Optional[str]:
        """
        为整个事件生成总结性分析
        """
        if not predictions:
            return None

        # 构建提示词
        prompt = f"""
        请用中文分析以下事件的可能性：
        事件：“{event}”
        
        历史预测记录：
        {self._format_predictions(predictions)}
        
        要求：
        - 控制在200字以内
        - 只输出分析内容，不带任何格式
        """
        return self._call_api(prompt)

    def generate_reasoning_single(self, event: str, probability: int) -> Optional[str]:
        """
        为单次预测生成分析
        """
        prompt = f"""
        请用中文简要分析以下事件的可能性：
        事件：“{event}”
        预测概率：{probability}%
        
        要求：
        - 控制在150字以内
        - 只输出分析内容，不带任何格式
        """
        return self._call_api(prompt)

    def _call_api(self, prompt: str) -> Optional[str]:
        """
        调用 Perplexity API 获取文本生成结果
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "你是一个专业的事件分析AI，输出简洁易懂的解释"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 200
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=15)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return content.strip()
            else:
                print(f"API请求失败: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"调用API出错: {e}")
            return None

    def _format_predictions(self, predictions: list) -> str:
        """
        格式化历史预测数据用于提示词
        """
        lines = []
        for pred in predictions:
            time_str = pred['timestamp'].strftime("%Y-%m-%d %H:%M") if 'timestamp' in pred else "未知时间"
            lines.append(f"- 时间: {time_str} | 概率: {pred['probability']}%")
        return "\n".join(lines)