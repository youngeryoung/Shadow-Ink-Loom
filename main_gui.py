# =============================================================
# [Shadow-Ink Loom]  
# Author: 烛鵼 Young 
# “For the Shadow-bird to mend the world, 
#     it first needs to see in black and white.”
# =============================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import img_processor  # 导入后端

class PCBToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片二值化处理工具")
        self.root.minsize(850, 700) # 设置最小尺寸，防止UI崩坏
        
        # --- 状态变量 ---
        self.src_image = None
        self.processed_image = None
        self.src_ratio = 1.0
        self.is_locked = True
        self.img_path = ""

        # --- 全局样式 ---
        self.setup_styles()

        # --- 核心容器 (垂直布局) ---
        self.main_container = tk.Frame(root, padx=10, pady=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 1. 顶部预览区 (2:3 比例)
        self.create_preview_area()

        # 2. 文件路径区 (分两行)
        self.create_file_area()

        # 3. 参数与设置区 (弹性流式布局)
        self.create_settings_area()

        # 4. 建议信息栏
        self.lbl_pixel_info = tk.Label(self.main_container, text="请先加载图片", fg="#2196F3", font=("Arial", 10, "bold"))
        self.lbl_pixel_info.pack(fill=tk.X, pady=5)

        # 5. 底部操作区
        self.create_action_area()

        # 6. 绑定窗口大小改变事件，用于自动重绘预览图
        self.root.bind("<Configure>", self.on_window_resize)
        self._resize_timer = None # 用于防抖

    def setup_styles(self):
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("Header.TLabel", font=("Arial", 10, "bold"))

    def create_preview_area(self):
        # 使用权重实现 2:3 比例
        self.preview_frame = tk.Frame(self.main_container, height=400)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_frame.columnconfigure(0, weight=2) # 左侧占比 2
        self.preview_frame.columnconfigure(1, weight=3) # 右侧占比 3
        self.preview_frame.rowconfigure(0, weight=1)

        # 左侧原图容器
        self.box_left = tk.LabelFrame(self.preview_frame, text="原始图片 (原比例) ", bg="#f0f0f0")
        self.box_left.grid(row=0, column=0, sticky="nsew", padx=2)
        self.panel_left = tk.Label(self.box_left, bg="#f0f0f0")
        self.panel_left.pack(fill=tk.BOTH, expand=True)

        # 右侧预览图容器
        self.box_right = tk.LabelFrame(self.preview_frame, text="生成预览 (二值化) ", bg="#f0f0f0")
        self.box_right.grid(row=0, column=1, sticky="nsew", padx=2)
        self.panel_right = tk.Label(self.box_right, bg="#f0f0f0")
        self.panel_right.pack(fill=tk.BOTH, expand=True)

    def create_file_area(self):
        file_frame = tk.LabelFrame(self.main_container, text=" 路径选择 ", padx=10, pady=5)
        file_frame.pack(fill=tk.X, pady=5)

        # 输入行
        in_row = tk.Frame(file_frame)
        in_row.pack(fill=tk.X, pady=2)
        tk.Label(in_row, text="输入图片:", width=10, anchor="w").pack(side=tk.LEFT)
        self.entry_input = tk.Entry(in_row)
        self.entry_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(in_row, text="选择图片", command=self.select_file, width=10).pack(side=tk.RIGHT)

        # 输出行
        out_row = tk.Frame(file_frame)
        out_row.pack(fill=tk.X, pady=2)
        tk.Label(out_row, text="输出目录:", width=10, anchor="w").pack(side=tk.LEFT)
        self.entry_output = tk.Entry(out_row)
        self.entry_output.insert(0, os.getcwd())
        self.entry_output.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(out_row, text="更改目录", command=self.select_output_dir, width=10).pack(side=tk.RIGHT)

    def create_settings_area(self):
        # 参数大容器（允许内部组件在宽度不足时“看似”换行，实则分组）
        settings_frame = tk.Frame(self.main_container)
        settings_frame.pack(fill=tk.X, pady=5)

        # 组1：丝印精度
        g1 = tk.LabelFrame(settings_frame, text=" 工艺细节 ", padx=5, pady=5)
        g1.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        tk.Label(g1, text="精度 (mm/px):").pack(side=tk.LEFT)
        self.var_precision = tk.DoubleVar(value=0.15)
        self.entry_precision = tk.Entry(g1, textvariable=self.var_precision, width=6)
        self.entry_precision.pack(side=tk.LEFT, padx=5)
        self.entry_precision.bind("<KeyRelease>", lambda e: self.update_pixel_info())

        # 组2：物理尺寸控制
        g2 = tk.LabelFrame(settings_frame, text=" 目标物理尺寸 ", padx=5, pady=5)
        g2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Label(g2, text="高:").pack(side=tk.LEFT)
        self.var_height_mm = tk.DoubleVar(value=80.0)
        self.entry_h = tk.Entry(g2, textvariable=self.var_height_mm, width=6)
        self.entry_h.pack(side=tk.LEFT, padx=2)
        self.entry_h.bind("<KeyRelease>", lambda e: self.on_dimension_change('h'))

        self.btn_lock = tk.Button(g2, text="🔒", command=self.toggle_lock, width=1, relief="flat")
        self.btn_lock.pack(side=tk.LEFT, padx=5)

        tk.Label(g2, text="宽:").pack(side=tk.LEFT)
        self.var_width_mm = tk.DoubleVar(value=0.0)
        self.entry_w = tk.Entry(g2, textvariable=self.var_width_mm, width=6)
        self.entry_w.pack(side=tk.LEFT, padx=1)
        self.entry_w.bind("<KeyRelease>", lambda e: self.on_dimension_change('w'))
        tk.Label(g2, text="mm").pack(side=tk.LEFT)

        # 组3：高级开关
        g3 = tk.LabelFrame(settings_frame, text=" 效果选项 ", padx=10, pady=5)
        g3.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.var_dither = tk.BooleanVar(value=True)
        tk.Checkbutton(g3, text="抖动", variable=self.var_dither, command=self.update_preview).pack(side=tk.LEFT)
        tk.Label(g3, text="描边:").pack(side=tk.LEFT, padx=(5,0))
        self.var_thickness = tk.IntVar(value=0)
        tk.Spinbox(g3, from_=0, to=100, textvariable=self.var_thickness, width=3, command=self.update_preview).pack(side=tk.LEFT)

    def create_action_area(self):
        action_frame = tk.Frame(self.main_container)
        action_frame.pack(fill=tk.X, pady=10)

        tk.Button(action_frame, text="🔄 刷新预览", command=self.update_preview, bg="#f8f9fa", height=2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(action_frame, text="💾 生成并保存图片", command=self.save_image, bg="#28a745", fg="white", font=("Arial", 10, "bold"), height=2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    # --- 核心逻辑逻辑 ---

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, path)
            self.img_path = path
            self.load_image()

    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, path)

    def load_image(self):
        img = cv2.imread(self.img_path)
        if img is None: return
        self.src_image = img
        h, w = img.shape[:2]
        self.src_ratio = w / h
        
        # 初始高度80mm，计算宽度
        h_mm = self.var_height_mm.get()
        self.var_width_mm.set(round(h_mm * self.src_ratio, 2))
        
        self.update_preview()

    def on_window_resize(self, event):
        # 只响应主窗口的尺寸变化，忽略控件自身的尺寸变化
        if event.widget != self.root:
            return
            
        # 如果正在连续调整大小（拖动中），先取消之前的任务
        if self._resize_timer is not None:
            self.root.after_cancel(self._resize_timer)
        
        # 延迟 150ms 执行真正的重绘任务
        self._resize_timer = self.root.after(150, self.perform_resize_render)

    def perform_resize_render(self):
        """真正执行缩略图重绘的任务"""
        if self.src_image is not None:
            self.display_image(self.src_image, self.panel_left)
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.panel_right)

    def display_image(self, cv_img, panel_widget):
        if cv_img is None: return
        
        # 强制强制刷新窗口布局状态，获取最新真实的容器大小
        panel_widget.update_idletasks()
        
        p_w = panel_widget.winfo_width()
        p_h = panel_widget.winfo_height()
        
        # 如果容器还没被渲染或太小，跳过以免 thumbnail 报错
        if p_w < 20 or p_h < 20: return

        if len(cv_img.shape) == 3:
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            cv_img_rgb = cv_img # 已经是单通道灰度
        
        img_pil = Image.fromarray(cv_img_rgb)
        
        # 保持比例计算缩略图
        # 注意：这里减去 4 像素作为安全边距，防止因边框导致的递归触发
        try:
            img_pil.thumbnail((p_w - 4, p_h - 4), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            panel_widget.config(image=img_tk)
            panel_widget.image = img_tk
        except Exception as e:
            print(f"Thumbnail error: {e}")

    def on_dimension_change(self, source):
        if not self.is_locked or self.src_image is None: 
            self.update_pixel_info()
            return
        try:
            if source == 'h':
                val = self.var_height_mm.get()
                self.var_width_mm.set(round(val * self.src_ratio, 3))
            else:
                val = self.var_width_mm.get()
                self.var_height_mm.set(round(val / self.src_ratio, 3))
            self.update_pixel_info()
        except: pass

    def toggle_lock(self):
        self.is_locked = not self.is_locked
        self.btn_lock.config(text="🔒" if self.is_locked else "🔓", fg="green" if self.is_locked else "red")
        if self.is_locked: # 重新锁定时更新比例
            try: self.src_ratio = self.var_width_mm.get() / self.var_height_mm.get()
            except: pass

    def update_pixel_info(self):
        try:
            pw = self.var_width_mm.get()
            ph = self.var_height_mm.get()
            prec = self.var_precision.get()
            px_w = int(round(pw / prec))
            px_h = int(round(ph / prec))
            self.lbl_pixel_info.config(text=f"📊 建议画布: {px_w} x {px_h} px  |  实际输出尺寸: {px_w*prec:.3f} x {px_h*prec:.3f} mm")
        except: pass

    def update_preview(self):
        if self.src_image is None: return
        try:
            prec = self.var_precision.get()
            px_w = int(round(self.var_width_mm.get() / prec))
            px_h = int(round(self.var_height_mm.get() / prec))
            
            self.processed_image = img_processor.process_image(
                self.src_image, px_w, px_h, 
                use_dithering=self.var_dither.get(),
                line_thickness=self.var_thickness.get()
            )
            self.display_image(self.src_image, self.panel_left)
            self.display_image(self.processed_image, self.panel_right)
            self.update_pixel_info()
        except Exception as e:
            print(f"预览失败: {e}")

    def save_image(self):
        # 1. 基础检查
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        
        # 2. 保存前强制按照当前 UI 参数重新处理图像，不使用缓存
        try:
            # 获取当前最新的参数
            prec = self.var_precision.get()
            mm_w = self.var_width_mm.get()
            mm_h = self.var_height_mm.get()
            
            px_w = int(round(mm_w / prec))
            px_h = int(round(mm_h / prec))
            
            # 执行算法（确保这是最新的结果）
            self.processed_image = img_processor.process_image(
                self.src_image, px_w, px_h, 
                use_dithering=self.var_dither.get(),
                line_thickness=self.var_thickness.get()
            )
            
            # 同步刷新预览区域，让用户知道保存的是哪张图
            self.display_image(self.processed_image, self.panel_right)
            self.update_pixel_info()
            
        except Exception as e:
            messagebox.showerror("生成失败", f"处理图像时发生错误: {e}")
            return

        # 3. 处理文件名
        try:
            # 动态检测精度字符串中的小数位数
            prec_str = self.entry_precision.get().strip()
            decimals = len(prec_str.split(".")[1]) if "." in prec_str else 0
        except:
            decimals = 2

        out_dir = self.entry_output.get()
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建目录: {e}")
                return

        # 获取不带后缀的文件名
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        
        # 格式化物理尺寸字符串
        w_str = f"{mm_w:.{decimals}f}"
        h_str = f"{mm_h:.{decimals}f}"
        
        mode = "dither" if self.var_dither.get() else "flat"
        
        # 构造最终文件名: name_20.00x30.00mm_dither.png
        out_filename = f"{base_name}_{w_str}x{h_str}mm_{mode}.png"
        out_path = os.path.join(out_dir, out_filename)
        
        # 4. 写入文件
        try:
            cv2.imwrite(out_path, self.processed_image)
            messagebox.showinfo("保存成功", f"文件已生成并保存至:\n{out_path}")
        except Exception as e:
            messagebox.showerror("写入失败", f"无法写入文件: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PCBToolApp(root)
    root.mainloop()
