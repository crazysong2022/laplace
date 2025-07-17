"""
动态事件概率预测系统 - 最终修正版 + AI Reasoning 支持（每条预测有独立分析）
"""
import logging
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import plotly.express as px
import psycopg2
import requests
import streamlit as st
from dotenv import load_dotenv
from psycopg2.extras import DictCursor

# 自定义模块
from comments import init_comments_table, display_comments_section
from reasoning import ReasoningService
import create_events

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
class Config:
    PAGE_TITLE = "🧠 动态事件概率预测系统"
    PAGE_LAYOUT = "wide"
    API_URL = "https://api.perplexity.ai/chat/completions"
    EVENTS_PER_PAGE = 100
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 15


# ------------------------------
# 数据库服务类
# ------------------------------
class DatabaseService:
    """封装数据库操作"""

    def __init__(self):
        self.conn = None
        self.connect()
        self.init_tables()

    def connect(self) -> bool:
        """建立数据库连接"""
        try:
            self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            st.error("无法连接到数据库，请检查配置")
            return False

    def init_tables(self):
        """初始化数据库表结构"""
        if not self.conn:
            return
        try:
            with self.conn.cursor() as cur:
                # events 表：移除了 reasoning 字段
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id SERIAL PRIMARY KEY,
                        event_text TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # predictions 表：新增 reasoning 字段
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        event_id INTEGER REFERENCES events(id),
                        probability INTEGER,
                        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        reasoning TEXT
                    )
                """)
                self.conn.commit()
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            st.error("数据库表初始化失败")

    def save_prediction(self, event_text: str, probability: int, reasoning: str = None) -> bool:
        """保存预测结果 + 单条分析"""
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cur:
                # 插入或获取事件ID
                cur.execute(
                    """
                    INSERT INTO events (event_text) 
                    VALUES (%s) 
                    ON CONFLICT (event_text) DO UPDATE SET event_text=EXCLUDED.event_text
                    RETURNING id
                    """,
                    (event_text,)
                )
                event_id = cur.fetchone()[0]
                # 插入预测结果和分析
                cur.execute(
                    "INSERT INTO predictions (event_id, probability, reasoning) VALUES (%s, %s, %s)",
                    (event_id, probability, reasoning)
                )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"保存预测结果失败: {str(e)}")
            self.conn.rollback()
            return False

    def get_predictions(self, event_text: str) -> pd.DataFrame:
        """获取指定事件的预测历史"""
        if not self.conn:
            return pd.DataFrame()
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT p.id, p.probability, p.predicted_at, p.reasoning
                    FROM predictions p
                    JOIN events e ON p.event_id = e.id
                    WHERE e.event_text = %s
                    ORDER BY p.predicted_at DESC
                """, (event_text,))
                rows = cur.fetchall()
                return pd.DataFrame(rows, columns=['id', 'probability', 'timestamp', 'reasoning']) if rows else pd.DataFrame()
        except Exception as e:
            logger.error(f"查询预测记录失败: {str(e)}")
            return pd.DataFrame()

    def get_recent_events(self, limit: int = 1000) -> List[str]:
        """获取最近的事件列表"""
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT event_text FROM events 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"获取事件列表失败: {str(e)}")
            return []

    def get_prediction_reasoning(self, prediction_id: int) -> Optional[str]:
        """获取某条预测的分析"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT reasoning FROM predictions WHERE id = %s", (prediction_id,))
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"查询reasoning失败: {str(e)}")
            return None

    def update_prediction_reasoning(self, prediction_id: int, reasoning: str) -> bool:
        """更新某条预测的reasoning"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE predictions SET reasoning = %s WHERE id = %s",
                    (reasoning, prediction_id)
                )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"更新reasoning失败: {str(e)}")
            self.conn.rollback()
            return False

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None


# ------------------------------
# AI预测服务类
# ------------------------------
class PredictionService:
    """处理AI预测相关逻辑"""

    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.db_service = DatabaseService()

    @staticmethod
    def clean_event_text(text: str) -> str:
        """清理事件文本"""
        if not isinstance(text, str):
            return ""
        text = text.replace("，", ",").replace("。", ".").replace("；", ";")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_prediction(self, event_question: str) -> Optional[int]:
        """获取AI预测结果"""
        cleaned_event = self.clean_event_text(event_question)
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = {
          "role": "user",
          "content": f"""请根据互联网上关于以下事件的最新新闻和相关信息，评估其在当前时间（{current_date}）发生的真实概率（0-100之间的整数）。如果你找不到确切信息，请说明原因。

        事件描述:
        "{cleaned_event}"

        要求：
        1. 优先参考最近的信息。
        2. 如果有多个来源冲突，请综合分析后给出最终概率。
        3. 你的输出必须是一个0到100之间的整数，不要添加任何额外文字或解释。
        """
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "你是一个专业的事件预测AI，只返回0-100的整数"},
                prompt
            ],
            "temperature": 0.3,
            "max_tokens": 10
        }
        try:
            response = requests.post(
                Config.API_URL,
                json=payload,
                headers=headers,
                timeout=Config.REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                match = re.search(r'\b\d{1,3}\b', content)
                if match:
                    probability = int(match.group())
                    if 0 <= probability <= 100:
                        return probability
                    logger.warning(f"概率值超出范围: {probability}")
                else:
                    if any(phrase in content for phrase in ["无法判断", "无法评估", "不适合"]):
                        st.info("⚠️ 该问题无法做出概率判断，请尝试其他事件。")
                    else:
                        logger.warning(f"无法提取概率值: {content}")
            else:
                logger.error(f"API请求失败: {response.status_code}")
                st.error("预测服务暂时不可用，请稍后再试")
        except requests.exceptions.Timeout:
            logger.error("API请求超时")
            st.error("请求超时，请稍后再试")
        except Exception as e:
            logger.error(f"请求异常: {str(e)}")
            st.error("预测过程中发生错误")
        return None

    def save_prediction(self, event_text: str, probability: int, reasoning: str = None) -> bool:
        """保存预测结果"""
        return self.db_service.save_prediction(event_text, probability, reasoning)

    def get_prediction_history(self, event_text: str) -> pd.DataFrame:
        """获取预测历史"""
        return self.db_service.get_predictions(event_text)


# ------------------------------
# UI服务类
# ------------------------------
class UIService:
    """处理用户界面相关逻辑"""

    def __init__(self):
        self.prediction_service = PredictionService()
        self.init_session_state()
        self.setup_page_config()
        self.apply_custom_styles()
        init_comments_table()

    def init_session_state(self):
        """初始化会话状态"""
        session_defaults = {
            "current_event": "",
            "new_event_input": "",
            "events_cache": self.prediction_service.db_service.get_recent_events(),
            "refresh_cache": False,
            "event_page": 1,
            "show_event_list": True,  # 新增：控制事件列表显示
        }
        for key, value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            layout=Config.PAGE_LAYOUT
        )

    def apply_custom_styles(self):
        """应用自定义CSS样式"""
        st.markdown("""
            <style>
                .main {
                    background-color: #f8f9fa;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                .predict-button {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .predict-button:hover {
                    background-color: #0056b3;
                    transform: translateY(-1px);
                }
                .footer {
                    text-align: center;
                    font-size: 0.9em;
                    color: gray;
                    margin-top: 50px;
                    padding: 15px 0;
                }
                .info-card {
                    background-color: #e9ecef;
                    padding: 15px;
                    border-left: 5px solid #007bff;
                    margin-bottom: 15px;
                    border-radius: 5px;
                }
                .event-card {
                    background-color: white;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: all 0.2s ease;
                }
                .event-card:hover {
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                    transform: translateY(-2px);
                }
                .pagination-info {
                    text-align: center;
                    margin: 10px 0;
                    color: #6c757d;
                }
                .event-list-container {
                    max-height: 400px;
                    overflow-y: auto;
                    margin: 15px 0;
                    padding: 10px;
                    border: 1px solid #eee;
                    border-radius: 8px;
                }
            </style>
        """, unsafe_allow_html=True)

    def update_event_input(self, event: str):
        """更新事件输入状态 - 修改版避免widget冲突"""
    # 先清除现有的输入widget
        if "event_input" in st.session_state:
          del st.session_state.event_input
    
    # 更新状态
        st.session_state.current_event = event
        st.session_state.show_event_list = False  # 选择事件后隐藏列表
    
    # 设置新的输入值
        st.session_state.event_input = event
        st.rerun()

    def _get_filtered_events(self, search_term: str) -> List[str]:
        """获取过滤后的事件列表"""
        if not search_term:
            return st.session_state.events_cache
        return [e for e in st.session_state.events_cache
                if search_term.lower() in e.lower()]

    def _get_paginated_events(self, events: List[str]) -> List[str]:
        """获取当前页的事件"""
        per_page = Config.EVENTS_PER_PAGE
        start_idx = (st.session_state.event_page - 1) * per_page
        end_idx = start_idx + per_page
        return events[start_idx:end_idx]

    def _render_pagination_controls(self, total_events: List[str]):
        """渲染分页控件"""
        total_pages = max(1, (len(total_events) + Config.EVENTS_PER_PAGE - 1) // Config.EVENTS_PER_PAGE)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "⬅️ 上一页",
                disabled=st.session_state.event_page <= 1,
                use_container_width=True
            ):
                st.session_state.event_page -= 1
                st.rerun()
        with col2:
            if st.button(
                "下一页 ➡️",
                disabled=st.session_state.event_page >= total_pages,
                use_container_width=True
            ):
                st.session_state.event_page += 1
                st.rerun()
        st.markdown(
            f'<div class="pagination-info">第 {st.session_state.event_page} 页 / 共 {total_pages} 页</div>',
            unsafe_allow_html=True
        )

    def render_event_list(self):
        """渲染事件列表（主页面版）"""
        with st.expander("📚 历史事件列表", expanded=True):  # 固定为 True，默认展开
           search_term = st.text_input("搜索历史事件", key="main_search", 
                                  placeholder="输入关键词筛选事件")
        
           matched_events = self._get_filtered_events(search_term)
           page_events = self._get_paginated_events(matched_events)
        
           if page_events:
            st.info("点击事件查看详细分析")
            for idx, event in enumerate(page_events):
                # 使用事件内容和索引创建唯一key
                btn_key = f"event_btn_{idx}_{hash(event)}"
                if st.button(
                    event,
                    key=btn_key,
                    use_container_width=True,
                    help="点击查看该事件的预测历史"
                ):
                    self.update_event_input(event)
                    st.rerun()
            
            # 分页控件
            self._render_pagination_controls(matched_events)
           else:
            st.info("未找到匹配的事件")
    def render_prediction_chart(self, predictions_df: pd.DataFrame):
        """渲染预测趋势图"""
        fig = px.line(
            predictions_df,
            x="timestamp",
            y="probability",
            labels={"probability": "发生概率(%)", "timestamp": "时间"},
            markers=True,
            line_shape="linear"
        )
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="预测时间",
            yaxis_title="发生概率(%)",
            yaxis_range=[0, 100],
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="red",
            annotation_text="50%基准线",
            annotation_position="bottom right"
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_latest_prediction(self, predictions_df: pd.DataFrame):
        """渲染最新预测结果"""
        latest_pred = predictions_df.iloc[0]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                "最新预测概率",
                f"{latest_pred['probability']}%",
                delta=f"更新时间: {latest_pred['timestamp'].strftime('%Y-%m-%d %H:%M')}"
            )
        with col2:
            progress_value = latest_pred['probability'] / 100
            st.progress(
                progress_value,
                text=f"发生可能性: {latest_pred['probability']}%"
            )

    def render_prediction_input(self):
       """渲染预测输入区域 + AI生成事件建议"""
       event_input = st.text_input(
        "输入您想预测的事件",
        value=st.session_state.get("new_event_input", ""),
        key="new_event_input",
        placeholder="例如：'2028年特朗普再次当选美国总统的可能性'"
       )

    # AI生成事件建议
       with st.expander("🔍 生成预测事件建议（推荐）"):
        categories = create_events.load_categories()

        # 国家选择
        country = st.selectbox("选择国家或地区", options=categories["countries"], key="country_selector")

        # 大类选择
        market_options = list(categories["market_categories"].keys())
        market = st.selectbox("选择预测市场大类", options=market_options, key="market_selector")

        # 小类选择
        subcategory_options = categories["market_categories"][market]
        subcategory = st.selectbox("选择预测市场小类", options=subcategory_options, key="subcategory_selector")

        if st.button("生成事件建议", use_container_width=True):
            with st.spinner("正在生成事件建议..."):
                suggested_events = create_events.generate_suggested_events(country, market, subcategory)
                if suggested_events:
                    st.session_state.suggested_events = suggested_events
                else:
                    st.warning("未能生成事件，请稍后再试")

        if "suggested_events" in st.session_state:
            selected_event = st.selectbox("从建议中选择一个事件", options=st.session_state.suggested_events)
            if st.button("使用该事件", use_container_width=True):
                st.session_state.new_event_input = selected_event
                st.session_state.current_event = selected_event
                st.rerun()

    # 提交预测按钮
       if st.button(
        "🚀 执行预测",
        key="predict_button",
        use_container_width=True,
        type="primary"
       ):
        if not event_input or not event_input.strip():
            st.warning("请输入有效的预测事件内容")
        else:
            self.handle_prediction_request(event_input)

    # 切换历史事件列表显示
       if st.button(
        "📚 显示历史事件列表" if not st.session_state.show_event_list else "❌ 隐藏历史事件列表",
        key="toggle_event_list",
        use_container_width=True
       ):
        st.session_state.show_event_list = not st.session_state.show_event_list
        st.rerun()

    def handle_prediction_request(self, event_text: str):
        """处理预测请求"""
        with st.spinner("正在分析预测..."):
            probability = self.prediction_service.get_prediction(event_text)
            if probability is not None:
                with st.spinner("生成本次预测的分析..."):
                    rs = ReasoningService()
                    new_reasoning = rs.generate_reasoning_single(event_text, probability)
                    success = self.prediction_service.save_prediction(event_text, probability, new_reasoning)
                if success:
                    st.session_state.current_event = event_text
                    st.success(f"✅ 预测成功！概率为：{probability}%")
                    st.session_state.refresh_cache = True
                    st.session_state.events_cache = self.prediction_service.db_service.get_recent_events()
                    st.rerun()
                else:
                    st.error("保存预测结果失败")

    def render_event_details(self):
       """渲染事件详情区域（含分页的历史预测分析）"""
       current_event = st.session_state.current_event
       predictions_df = self.prediction_service.get_prediction_history(current_event)

       if not predictions_df.empty:
        st.subheader(f"📌 事件分析: {current_event}")
        
        # 趋势图
        self.render_prediction_chart(predictions_df)
        
        # 最新预测值
        self.render_latest_prediction(predictions_df)

        st.markdown("### 🧠 历史预测分析")

        # 初始化 session_state 中的页码
        if "page_num" not in st.session_state:
            st.session_state.page_num = 1

        predictions_list = predictions_df.to_dict(orient="records")
        total_predictions = len(predictions_list)
        items_per_page = 5
        total_pages = (total_predictions + items_per_page - 1) // items_per_page

        # 分页数据
        start_idx = (st.session_state.page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_data = predictions_list[start_idx:end_idx]

        # 分页控件 - 添加唯一key
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ 上一页", 
                       disabled=st.session_state.page_num == 1, 
                       use_container_width=True,
                       key=f"prev_page_{st.session_state.page_num}"):
                st.session_state.page_num -= 1
                st.rerun()
        with col3:
            if st.button("下一页 ➡️", 
                       disabled=st.session_state.page_num >= total_pages, 
                       use_container_width=True,
                       key=f"next_page_{st.session_state.page_num}"):
                st.session_state.page_num += 1
                st.rerun()
        with col2:
            st.markdown(
                f"<div style='text-align:center; margin-top:8px;'>第 {st.session_state.page_num} 页 / 共 {total_pages} 页</div>",
                unsafe_allow_html=True,
            )

        # 展示当前页的预测分析
        for idx, pred in enumerate(page_data):
            with st.expander(f"📅 {pred['timestamp'].strftime('%Y-%m-%d %H:%M')} | {pred['probability']}%"):
                db_reasoning = pred['reasoning']
                if db_reasoning:
                    st.markdown(db_reasoning)
                else:
                    with st.spinner("正在生成本次预测的分析..."):
                        rs = ReasoningService()
                        new_reasoning = rs.generate_reasoning_single(current_event, pred['probability'])
                        if new_reasoning:
                            self.prediction_service.db_service.update_prediction_reasoning(pred["id"], new_reasoning)
                            st.markdown(new_reasoning)
                        else:
                            st.warning("无法生成本次预测的分析")

        # 留言系统
        display_comments_section(current_event)

       else:
        st.info("暂无该事件的预测记录")

    def render_main_content(self):
       """渲染主内容区域"""
       st.title(Config.PAGE_TITLE)
    
    # 信息卡片
       st.markdown('''
        <div class="info-card">
            <strong>使用说明:</strong> 
            1. 输入您关心的事件，系统将分析其发生的概率(0-100%)
            2. 点击"显示历史事件列表"查看过往预测
            3. 点击事件可查看详细分析
        </div>
    ''', unsafe_allow_html=True)
    
    # 预测输入区域
       self.render_prediction_input()
    
    # 显示/隐藏事件列表
       if st.session_state.show_event_list:
        self.render_event_list()
        # 如果正在显示事件列表，清空当前事件
        st.session_state.current_event = ""
       elif st.session_state.current_event:
        # 如果选择了事件，显示详情
        self.render_event_details()
       else:
        st.info("请输入一个新事件或点击\"显示历史事件列表\"选择历史事件")

# ------------------------------
# 主程序
# ------------------------------
def main():
    """主应用程序入口"""
    ui_service = UIService()
    ui_service.render_main_content()


if __name__ == "__main__":
    main()