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
            self.ocr = RapidOCR()
            print("[INFO] RapidOCR 初始化成功")
        except Exception as e:
            print(f"[WARN] OCR初始化失败: {e}")
            self.ocr = None

    def recognize_text(self, img: np.ndarray) -> str:
        """识别图片中的文字"""
        if self.ocr is None:
            return "[OCR未初始化]"
        
        try:
            # 转换为PIL Image
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            pil_img = Image.fromarray(img_rgb)
            
            # OCR识别
            result, elapse = self.ocr(pil_img)
            
            if result:
                texts = [item[1] for item in result]
                return ' '.join(texts)
            else:
                return ""
        except Exception as e:
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
        """检测聊天消息"""
        h, w = chat_area.shape[:2]
        gray = cv2.cvtColor(chat_area, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.absolute(sobel_y)
        sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))
        _, binary = cv2.threshold(sobel_y, 30, 255, cv2.THRESH_BINARY)
        
        horizontal_lines = []
        for y in range(h):
            row_sum = np.sum(binary[y, :])
            if row_sum > w * 100:
                horizontal_lines.append(y)
        
        merged_lines = []
        if horizontal_lines:
            current_line = horizontal_lines[0]
            for line in horizontal_lines[1:]:
                if line - current_line > 5:
                    merged_lines.append(current_line)
                    current_line = line
            merged_lines.append(current_line)
        
        messages = []
        if len(merged_lines) >= 2:
            for i in range(len(merged_lines) - 1):
                y1 = merged_lines[i]
                y2 = merged_lines[i + 1]
                height = y2 - y1
                if 20 < height < 150:
                    region = chat_area[y1:y2, :]
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    left_avg = np.mean(region_gray[:, :int(w * 0.4)])
                    right_avg = np.mean(region_gray[:, int(w * 0.6):])
                    is_my = right_avg > left_avg + 10
                    
                    _, region_thresh = cv2.threshold(region_gray, 240, 255, cv2.THRESH_BINARY_INV)
                    cols = np.sum(region_thresh, axis=0)
                    x_coords = np.where(cols > 0)[0]
                    
                    if len(x_coords) > 0:
                        x_start = max(0, x_coords[0] - 10)
                        x_end = min(w - 1, x_coords[-1] + 10)
                        width = x_end - x_start
                        
                        messages.append({
                            'x': x_start, 'y': y1,
                            'w': width, 'h': height,
                            'is_my': is_my
                        })
        
        if not messages:
            row_height = 30
            for y in range(0, h - row_height, row_height):
                row = chat_area[y:y + row_height, :]
                row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(row_gray)
                if 50 < avg_brightness < 245:
                    _, row_thresh = cv2.threshold(row_gray, 200, 255, cv2.THRESH_BINARY_INV)
                    cols = np.sum(row_thresh, axis=0)
                    x_coords = np.where(cols > 0)[0]
                    if len(x_coords) > 10:
                        x_start = x_coords[0]
                        x_end = x_coords[-1]
                        width = x_end - x_start + 1
                        is_my = x_start > w * 0.5
                        messages.append({
                            'x': x_start, 'y': y,
                            'w': width, 'h': row_height,
                            'is_my': is_my
                        })
        
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
        top_y = 100
        bottom_y = h - 120
        return img[top_y:bottom_y, left_x:w], left_x, top_y

    def extract_profile_info(self, img: np.ndarray) -> dict:
        """提取顶部个人信息区域"""
        h, w = img.shape[:2]
        return {
            'region': (int(w * 0.45), 20, int(w * 0.25), 80),
            'name_region': (int(w * 0.45) + 60, 20, 140, 40),
            'position_region': (int(w * 0.45) + 60, 60, 140, 25),
            'avatar_region': (int(w * 0.45) + 10, 20, 50, 50)
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
        
        # 2. 识别聊天区域消息
        print("[INFO] 识别聊天区域消息...")
        chat_area, chat_x, chat_y = self.extract_chat_area(img)
        messages = self.detect_chat_messages(chat_area)
        
        adjusted_messages = []
        for i, msg in enumerate(messages):
            # OCR识别消息内容
            msg_img = chat_area[msg['y']:msg['y'] + msg['h'], msg['x']:msg['x'] + msg['w']]
            msg_text = self.recognize_text(msg_img)
            
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
        
        # 3. 提取个人信息区域
        print("[INFO] 提取个人信息...")
        profile_info = self.extract_profile_info(img)
        
        # OCR识别姓名和职位
        name_img = img[profile_info['name_region'][1]:profile_info['name_region'][1] + profile_info['name_region'][3],
                       profile_info['name_region'][0]:profile_info['name_region'][0] + profile_info['name_region'][2]]
        profile_name = self.recognize_text(name_img)
        
        position_img = img[profile_info['position_region'][1]:profile_info['position_region'][1] + profile_info['position_region'][3],
                          profile_info['position_region'][0]:profile_info['position_region'][0] + profile_info['position_region'][2]]
        profile_position = self.recognize_text(position_img)
        
        result["profile"] = {
            "name": profile_name,
            "position": profile_position,
            "name_region": profile_info['name_region'],
            "position_region": profile_info['position_region'],
            "avatar_region": profile_info['avatar_region']
        }
        
        print(f"[INFO] 识别完成: {len(candidates)}个候选人, {len(adjusted_messages)}条消息")
        return result


def main():
    """主函数"""
    recognizer = OCRBossRecognizer()
    
    # 获取图片路径
    print("\n" + "=" * 60)
    print("BOSS直聘信息识别工具")
    print("=" * 60)
    print("\n请输入截图文件的完整路径")
    print("提示：可以直接拖拽图片文件到命令行窗口")
    print("（直接按回车使用默认路径: mmexport1781002649103.jpg）")
    
    img_path = input("\n图片路径: ").strip().strip('"').strip("'")
    
    # 如果用户直接按回车，使用默认路径
    if not img_path:
        img_path = "mmexport1781002649103.jpg"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(img_path):
        print(f"[ERROR] 文件不存在: {img_path}")
        print("请检查路径是否正确！")
        return
    
    print(f"\n[INFO] 正在识别图片: {img_path}")
    
    result = recognizer.recognize(img_path)
    
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return
    
    # 打印结果
    print("\n" + "=" * 60)
    print("BOSS直聘信息识别报告（含OCR文字识别）")
    print("=" * 60)
    print(f"识别时间: {result['timestamp']}")
    print(f"图片尺寸: {result['image_size']['width']}x{result['image_size']['height']}")
    
    print("\n【个人信息】")
    print("-" * 40)
    print(f"姓名: {result['profile']['name']}")
    print(f"职位: {result['profile']['position']}")
    
    print("\n【左侧候选人列表】")
    print("-" * 40)
    print(f"候选人总数: {result['left_panel']['total_candidates']}")
    print(f"未读消息: {result['left_panel']['unread_count']}")
    print(f"已读消息: {result['left_panel']['total_candidates'] - result['left_panel']['unread_count']}")
    
    print("\n【候选人详情】")
    for c in result['left_panel']['candidates']:
        status = "🔴 未读" if c['is_unread'] else "⚪ 已读"
        print(f"  {status} 卡片#{c['index']}")
        print(f"    姓名: {c['name']}")
        print(f"    消息: {c['message_preview']}")
    
    print("\n【聊天区域消息】")
    print("-" * 40)
    print(f"消息总数: {result['chat_area']['message_count']}")
    print(f"我方消息: {result['chat_area']['my_messages']}")
    print(f"对方消息: {result['chat_area']['other_messages']}")
    
    print("\n【消息详情】")
    for msg in result['chat_area']['messages']:
        status = "💬" if msg['is_my'] else "👤"
        print(f"  {status} {msg['sender']}: {msg['content']}")
    
    # 保存JSON结果
    json_file = "ocr_recognition_result.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] JSON结果已保存: {json_file}")
    
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
    
    txt_file = "boss_ocr_report.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_content))
    print(f"[INFO] 详细报告已保存: {txt_file}")


if __name__ == "__main__":
    main()
