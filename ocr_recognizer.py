"""
带OCR文字识别的BOSS直聘信息识别工具
功能：识别左侧候选人列表 + 中间聊天框消息内容 + 文字识别
"""

import cv2
import numpy as np
import json
import time
from PIL import Image


class OCRBossRecognizer:
    """带OCR的识别器"""

    def __init__(self):
        # 未读红点颜色范围（HSV）
        self.unread_red_lower = np.array([0, 80, 80])
        self.unread_red_upper = np.array([15, 255, 255])
        self.unread_red_lower2 = np.array([160, 80, 80])
        self.unread_red_upper2 = np.array([180, 255, 255])
        
        # 初始化OCR
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        """初始化OCR引擎"""
        try:
            from rapidocr_onnxruntime import RapidOCR
            # 【关键修改】：启用高精度模型
            self.ocr = RapidOCR(
                det_model_name='ch_PP-OCRv4_det', 
                rec_model_name='ch_PP-OCRv4_rec'
            )
            print("[INFO] RapidOCR (V4 High Precision) 初始化成功")
        except Exception as e:
            print(f"[WARN] OCR初始化失败: {e}")
            self.ocr = None

    def recognize_text(self, img: np.ndarray) -> str:
        """识别图片中的文字"""
        if self.ocr is None:
            return "[OCR未初始化]"
        
        try:
            if img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5:
                return ""

            # 确保图像是 3 通道的
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # 调用 OCR
            result, elapse = self.ocr(img)
            
            if result:
                # result 结构: [ [box, text, score], ... ]
                # 提取所有文本并用空格连接
                texts = [item[1] for item in result if item[1]]
                return ' '.join(texts) 
            else:
                return ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[识别失败: {str(e)}]"

    def detect_unread_markers(self, img: np.ndarray) -> list:
        """检测未读红点标记"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.unread_red_lower, self.unread_red_upper)
        mask2 = cv2.inRange(hsv, self.unread_red_lower2, self.unread_red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if 8 < area < 400 and 0.4 < cw / ch < 2.5:
                markers.append((x, y, cw, ch))
        return markers

    def detect_card_regions(self, img: np.ndarray) -> list:
        """检测候选人卡片区域"""
        h, w = img.shape[:2]
        card_height = 90
        cards = []
        for y in range(80, h - card_height, card_height):
            cards.append((0, y, w, card_height))
        return cards

    def detect_chat_messages(self, chat_area: np.ndarray) -> list:
        """
        检测聊天消息区域
        改进：
        1. 使用形态学操作合并同一气泡内的多行文字。
        2. 增加内容密度过滤，去除空白噪点气泡。
        3. 增加宽度和位置过滤，避免误检上方信息栏。
        """
        h, w = chat_area.shape[:2]
        if h == 0 or w == 0:
            return []

        gray = cv2.cvtColor(chat_area, cv2.COLOR_BGR2GRAY)
        
        # 1. 二值化：提取深色文字
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. 形态学膨胀：连接垂直方向上相邻的文字行
        # 核宽度15：连接左右字符
        # 核高度7：连接上下行（比之前稍小，减少误连）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)) 
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 3. 寻找轮廓
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        messages = []
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # 过滤噪点：太小或太高的区域
            if cw < 15 or ch < 10 or ch > 300:
                continue
            
            # 【关键过滤】：检查原始二值图中该区域的内容密度
            roi_binary = binary[y:y+ch, x:x+cw]
            content_pixels = np.count_nonzero(roi_binary)
            total_pixels = cw * ch
            
            # 如果内容占比小于 5%，视为空白气泡
            if total_pixels > 0 and (content_pixels / total_pixels) < 0.05:
                continue
                
            # 【新增过滤】：宽度不能太宽，避免包含整个“沟通职位”条目
            # 通常消息气泡宽度不会超过屏幕的一半
            if cw > w * 0.8:
                continue
                
            # 【新增过滤】：Y坐标不能太靠上，避开顶部的个人信息栏
            # 假设个人信息栏高度约为 100px，我们从 Y=100 开始检测
            if y < 100:
                continue
            
            # 4. 判断是“我”发的还是“对方”发的
            center_x = x + cw / 2
            is_my = center_x > w * 0.5
            
            # 5. 增加 Padding，确保文字不被切断
            padding_y = 5
            padding_x = 5
            
            final_y = max(0, y - padding_y)
            final_h = min(h - final_y, ch + 2 * padding_y)
            final_x = max(0, x - padding_x)
            final_w = min(w - final_x, cw + 2 * padding_x)
            
            messages.append({
                'x': final_x,
                'y': final_y,
                'w': final_w,
                'h': final_h,
                'is_my': is_my
            })
        
        # 6. 按 Y 轴排序，确保消息顺序正确
        messages.sort(key=lambda m: m['y'])
        
        return messages

    def extract_left_panel(self, img: np.ndarray) -> tuple:
        """提取左侧列表面板"""
        h, w = img.shape[:2]
        left_x = int(w * 0.12) if w > 500 else 60
        right_x = int(w * 0.45) if w > 500 else 420
        return img[0:h, left_x:right_x], left_x

    def extract_chat_area(self, img: np.ndarray) -> tuple:
        """提取中间聊天区域"""
        h, w = img.shape[:2]
        left_x = int(w * 0.45) if w > 500 else 420
        top_y = 100  # 【关键修改】：从 Y=100 开始，避开顶部的个人信息栏
        bottom_y = h - 120
        return img[top_y:bottom_y, left_x:w], left_x, top_y

    def extract_profile_info(self, img: np.ndarray) -> dict:
        """智能提取个人信息区域（基于OCR全区域扫描）"""
        h, w = img.shape[:2]
        
        # 1. 提取顶部区域（Y=0 到 Y=200，覆盖所有个人信息）
        top_area = img[0:min(200, h), 0:w]
        
        # 2. 使用 OCR 扫描整个顶部区域
        text_result = self.recognize_text(top_area)
        
        # 3. 按行分割并清理
        lines = [line.strip() for line in text_result.split(' ') if line.strip()]
        
        # 4. 根据关键词和内容特征匹配
        name = ""
        position = ""
        job_title = ""
        expectation = ""
        
        # 遍历所有识别出的文本行
        for i, line in enumerate(lines):
            # 姓名识别：通常在最前面，可能包含"刚刚活跃"等状态
            if not name and len(line) <= 10:
                # 过滤掉纯数字或特殊字符
                if any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in line):
                    # 如果包含"刚刚"、"活跃"等词，提取前面的部分作为姓名
                    if "刚刚" in line or "活跃" in line or "在线" in line:
                        parts = line.split("刚刚")[0].split("活跃")[0].split("在线")[0].strip()
                        if parts and len(parts) <= 5:
                            name = parts
                    else:
                        # 可能是纯姓名
                        name = line
            
            # 职位/学历信息：包含年龄、学历等关键词
            elif not position and ("岁" in line or "应届" in line or "高中" in line or "本科" in line or "大专" in line):
                position = line
            
            # 沟通职位：通常较长，包含工作描述
            elif not job_title and ("日结" in line or "前台" in line or "自拍馆" in line or "包吃住" in line):
                job_title = line
            
            # 期望：包含地点、薪资等信息
            elif not expectation and ("·" in line and ("K" in line or "k" in line)):
                expectation = line
        
        return {
            'name': name,
            'position': position,
            'job_title': job_title,
            'expectation': expectation,
            'raw_text': text_result,  # 保留原始识别结果用于调试
            'region': (0, 0, w, min(200, h))
        }

    def recognize(self, img_path: str):
        """执行完整识别"""
        img = cv2.imread(img_path)
        if img is None:
            return {"error": f"无法加载图片: {img_path}"}
        
        h, w = img.shape[:2]
        print(f"[INFO] 图片尺寸: {w}x{h}")
        
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": {"width": w, "height": h},
            "left_panel": {},
            "chat_area": {},
            "profile": {}
        }
        
        # 1. 识别左侧候选人列表
        print("[INFO] 识别左侧候选人列表...")
        left_panel, panel_x = self.extract_left_panel(img)
        unread_markers = self.detect_unread_markers(left_panel)
        card_regions = self.detect_card_regions(left_panel)
        
        candidates = []
        unread_count = 0
        for i, (cx, cy, cw, ch) in enumerate(card_regions):
            has_unread = False
            for (mx, my, mw, mh) in unread_markers:
                if cy - 10 <= my <= cy + ch + 10:
                    has_unread = True
                    unread_count += 1
                    break
            
            # OCR识别姓名区域（卡片上半部分）
            name_region = left_panel[cy:cy + int(ch * 0.5), cx:cx + cw]
            name_text = self.recognize_text(name_region)
            
            # OCR识别消息预览（卡片下半部分）
            msg_region = left_panel[cy + int(ch * 0.5):cy + ch, cx:cx + cw]
            msg_text = self.recognize_text(msg_region)
            
            candidates.append({
                "index": int(i),
                "is_unread": bool(has_unread),
                "name": name_text,
                "message_preview": msg_text,
                "region": [int(panel_x + cx), int(cy), int(cw), int(ch)]
            })
        
        result["left_panel"] = {
            "total_candidates": int(len(candidates)),
            "unread_count": int(unread_count),
            "candidates": candidates,
            "markers": [[int(panel_x + mx), int(my), int(mw), int(mh)] for mx, my, mw, mh in unread_markers]
        }
        
        # 2. 提取顶部个人信息区域
        print("[INFO] 提取个人信息...")
        profile_info = self.extract_profile_info(img)
        
        # 【关键修改】：直接使用智能检测的结果，不再使用硬编码坐标
        result["profile"] = {
            "name": profile_info['name'],
            "position": profile_info['position'],
            "job_title": profile_info['job_title'],
            "expectation": profile_info['expectation'],
            "region": profile_info['region']
        }
        
        # 【调试信息】打印原始OCR识别结果
        if profile_info.get('raw_text'):
            print(f"[DEBUG] 个人信息区域原始OCR结果: {profile_info['raw_text']}")
        
        # 3. 识别聊天区域消息
        print("[INFO] 识别聊天区域消息...")
        chat_area, chat_x, chat_y = self.extract_chat_area(img)
        messages = self.detect_chat_messages(chat_area)
        
        adjusted_messages = []
        for i, msg in enumerate(messages):
            # 确保坐标不越界
            x1 = max(0, msg['x'])
            y1 = max(0, msg['y'])
            x2 = min(chat_area.shape[1], msg['x'] + msg['w'])
            y2 = min(chat_area.shape[0], msg['y'] + msg['h'])
            
            # 截取完整的消息气泡区域
            msg_img = chat_area[y1:y2, x1:x2]
            
            # OCR识别消息内容
            msg_text = self.recognize_text(msg_img)
            
            # 【可选】如果识别结果为空，可以选择跳过不加入列表
            if not msg_text.strip():
                continue

            adjusted_messages.append({
                "index": int(i),
                "sender": "我" if msg['is_my'] else "对方",
                "is_my": bool(msg['is_my']),
                "content": msg_text,
                "region": [int(chat_x + msg['x']), int(chat_y + msg['y']), int(msg['w']), int(msg['h'])]
            })
        
        result["chat_area"] = {
            "message_count": int(len(adjusted_messages)),
            "my_messages": int(sum(1 for m in adjusted_messages if m['is_my'])),
            "other_messages": int(sum(1 for m in adjusted_messages if not m['is_my'])),
            "messages": adjusted_messages
        }
        
        print(f"[INFO] 识别完成: {len(candidates)}个候选人, {len(adjusted_messages)}条消息")
        return result


def main():
    """主函数"""
    recognizer = OCRBossRecognizer()
    
    # 获取图片路径
    img_path = input("请输入截图路径(直接回车使用默认): ").strip().strip('"').strip("'")
    if not img_path:
        img_path = "mmexport1781002649103.jpg"
    
    import os
    if not os.path.exists(img_path):
        print(f"[ERROR] 文件不存在: {img_path}")
        return
    
    result = recognizer.recognize(img_path)
    
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return
    
    # 保存JSON结果
    with open("ocr_recognition_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 生成详细报告
    txt_content = []
    txt_content.append("=" * 60)
    txt_content.append("BOSS直聘信息识别完整报告（含OCR文字识别）")
    txt_content.append("=" * 60)
    txt_content.append(f"识别时间: {result['timestamp']}")
    txt_content.append(f"截图文件: {img_path}")
    txt_content.append(f"图片尺寸: {result['image_size']['width']} × {result['image_size']['height']}")
    txt_content.append("")
    
    # 个人信息
    txt_content.append("【一、个人信息】")
    txt_content.append("-" * 40)
    txt_content.append(f"姓名: {result['profile']['name']}")
    txt_content.append(f"职位: {result['profile']['position']}")
    txt_content.append(f"沟通职位: {result['profile'].get('job_title', '')}")
    txt_content.append(f"期望: {result['profile'].get('expectation', '')}")
    txt_content.append("")
    
    # 左侧列表
    txt_content.append("【二、左侧候选人列表】")
    txt_content.append("-" * 40)
    txt_content.append(f"📊 候选人总数: {result['left_panel']['total_candidates']}")
    txt_content.append(f"🔴 未读消息: {result['left_panel']['unread_count']}")
    txt_content.append(f"⚪ 已读消息: {result['left_panel']['total_candidates'] - result['left_panel']['unread_count']}")
    txt_content.append("")
    txt_content.append("候选人详情:")
    for c in result['left_panel']['candidates']:
        status = "[未读]" if c['is_unread'] else "[已读]"
        txt_content.append(f"  卡片#{c['index']}: {status}")
        txt_content.append(f"    姓名: {c['name']}")
        txt_content.append(f"    消息预览: {c['message_preview']}")
    txt_content.append("")
    
    # 聊天区域
    txt_content.append("【三、聊天区域消息】")
    txt_content.append("-" * 40)
    txt_content.append(f"📝 消息总数: {result['chat_area']['message_count']}")
    txt_content.append(f"💬 我方消息: {result['chat_area']['my_messages']}")
    txt_content.append(f"👤 对方消息: {result['chat_area']['other_messages']}")
    txt_content.append("")
    txt_content.append("消息详情:")
    for i, msg in enumerate(result['chat_area']['messages'], 1):
        sender = "我" if msg['is_my'] else "对方"
        txt_content.append(f"  [{i}] {sender}: {msg['content']}")
    txt_content.append("")
    
    txt_content.append("=" * 60)
    txt_content.append("识别完成！")
    txt_content.append("=" * 60)
    
    with open("boss_ocr_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_content))
    
    print(f"[INFO] 结果已保存: ocr_recognition_result.json, boss_ocr_report.txt")


if __name__ == "__main__":
    main()