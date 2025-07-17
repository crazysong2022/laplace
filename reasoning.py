import os
import re
import requests
from typing import Optional
from datetime import datetime

class ReasoningService:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai/chat/completions"

    def generate_reasoning_single(self, event: str, probability: int, created_at: datetime = None) -> Optional[str]:
        """
        为单次预测生成分析，AI会根据事件语言自动使用相同语言输出，并结合时间背景解释概率
        """
        time_context = self._format_time_context(created_at)

        prompt = f"""
        You are a professional analyst AI that answers in the same language as the input event.

        Event: "{event}"
        Predicted Probability: {probability}%
        Time Context: {time_context}

        Please analyze why this probability was given:
        1. Consider the context of the event and its time of creation
        2. Explain your reasoning clearly
        3. Output only the analysis, no formatting
        4. Use the SAME LANGUAGE as the event text
        5. Keep it under 150 words
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
                {"role": "system", "content": "You are a professional analyst AI that answers in the same language as the input event."},
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

    def _format_time_context(self, created_at: datetime) -> str:
        """格式化创建时间为自然语言描述"""
        if not created_at:
            return "unknown time"
        return created_at.strftime("%B %d, %Y")  # 英文格式，例如 "July 17, 2025"