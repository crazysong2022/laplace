# comments.py

import streamlit as st
import psycopg2
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 数据库连接配置
DATABASE_URL = os.getenv("DATABASE_URL")


def get_db_connection():
    """创建数据库连接"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"数据库连接失败: {str(e)}")
        return None


def init_comments_table():
    """初始化留言表（如果不存在）"""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS comments (
                        id SERIAL PRIMARY KEY,
                        event_id INTEGER REFERENCES events(id),
                        content TEXT NOT NULL,
                        parent_id INTEGER REFERENCES comments(id),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            st.error(f"留言表初始化失败: {str(e)}")
        finally:
            conn.close()


def get_event_id(event_text):
    """根据事件文本获取对应的event_id"""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM events WHERE event_text = %s", (event_text,))
            result = cur.fetchone()
            return result[0] if result else None
    finally:
        conn.close()


def save_comment(event_id, content, parent_id=None):
    """保存一条留言或回复"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO comments (event_id, content, parent_id) VALUES (%s, %s, %s)",
                (event_id, content, parent_id)
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"保存留言失败: {str(e)}")
        return False
    finally:
        conn.close()


def get_comments_by_event(event_id):
    """获取某个事件下的所有留言及回复"""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, parent_id, created_at 
                FROM comments 
                WHERE event_id = %s 
                ORDER BY created_at DESC
            """, (event_id,))
            rows = cur.fetchall()
            # 转换为字典列表
            return [
                {"id": r[0], "content": r[1], "parent_id": r[2], "created_at": r[3]}
                for r in rows
            ]
    finally:
        conn.close()


def display_comments_section(event_text):
    """展示留言区域的完整UI组件（供主程序调用）"""
    event_id = get_event_id(event_text)
    if event_id is None:
        st.warning("无法找到该事件的ID")
        return

    st.markdown("---")
    st.subheader("💬 匿名留言区")

    # 提交新留言
    new_comment = st.text_area("写点什么...", key="new_comment_input")
    if st.button("发送留言", key="send_new_comment"):
        if new_comment.strip():
            if save_comment(event_id, new_comment):
                st.success("留言成功！")
                st.rerun()
        else:
            st.warning("留言内容不能为空")

    # 获取留言数据
    comments = get_comments_by_event(event_id)

    if comments:
        # 构建留言树结构
        comment_tree = {}
        replies = []

        for c in comments:
            if c["parent_id"] is None:
                comment_tree[c["id"]] = c
            else:
                replies.append(c)

        # 绑定回复到主留言
        for reply in replies:
            if reply["parent_id"] in comment_tree:
                if "replies" not in comment_tree[reply["parent_id"]]:
                    comment_tree[reply["parent_id"]]["replies"] = []
                comment_tree[reply["parent_id"]]["replies"].append(reply)

        # 展示留言
        for cid, comment in comment_tree.items():
            st.markdown(f"""
                <div style="border-left: 3px solid #444; padding-left: 10px; margin-bottom: 15px;">
                    <small>匿名用户 · {comment['created_at'].strftime('%Y-%m-%d %H:%M')}</small><br>
                    <p>{comment['content']}</p>
                </div>
            """, unsafe_allow_html=True)

            # 回复按钮和输入框
            reply_key = f"reply_{cid}"
            reply_content = st.text_area("回复这条留言", key=reply_key)
            if st.button(f"回复 #{cid}", key=f"btn_reply_{cid}"):
                if reply_content.strip():
                    if save_comment(event_id, reply_content, parent_id=cid):
                        st.success("回复成功！")
                        st.rerun()
                else:
                    st.warning("回复内容不能为空")

            # 显示该留言的回复
            if "replies" in comment:
                for reply in comment["replies"]:
                    st.markdown(f"""
                        <div style="margin-left: 20px; border-left: 2px dashed gray; padding-left: 10px; margin-top: 5px;">
                            <small>匿名用户 · {reply['created_at'].strftime('%Y-%m-%d %H:%M')}</small><br>
                            <p>{reply['content']}</p>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("暂无留言")