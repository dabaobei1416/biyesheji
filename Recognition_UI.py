import smtplib
from email.mime.text import MIMEText
from email.header import Header
import random
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from QtFusion.path import abs_path
from QtFusion.utils import drawRectBox

from LoggerRes import ResultLogger, LogTable
from YOLOv8v5Model import YOLOv8v5Detector
from datasets.Violence.label_name import Label_list
from style_css import def_css_hitml
from utils_web import save_uploaded_file, concat_results, load_default_image, get_camera_names

import threading  # 导入线程模块
from playsound import playsound  # 导入playsound库

class Detection_UI:
    """
    检测系统类。

    Attributes:
        model_type (str): 模型类型。
        conf_threshold (float): 置信度阈值。
        iou_threshold (float): IOU阈值。
        selected_camera (str): 选定的摄像头。
        file_type (str): 文件类型。
        uploaded_file (FileUploader): 上传的文件。
        detection_result (str): 检测结果。
        detection_location (str): 检测位置。
        detection_confidence (str): 检测置信度。
        detection_time (str): 检测用时。
    """

    def __init__(self):
        """
        初始化行人跌倒检测系统的参数。
        """
        # 初始化类别标签列表和为每个类别随机分配颜色
        self.cls_name = Label_list  # 确保 Label_list 包含所有需要检测的目标类别
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.cls_name))]

        # 设置页面标题
        self.title = "校园学生异常行为分析系统"
        self.setup_page()  # 初始化页面布局
        def_css_hitml()  # 应用 CSS 样式

        # 初始化检测相关的配置参数
        self.model_type = None
        self.conf_threshold = 0.25  # 默认置信度阈值


        # 初始化相机和文件相关的变量
        self.selected_camera = None
        self.file_type = None
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file = None  # 自定义的模型文件

        # 初始化检测结果相关的变量
        self.detection_result = None
        self.detection_location = None
        self.detection_confidence = None
        self.detection_time = None

        # 初始化UI显示相关的变量
        self.display_mode = None  # 设置显示模式
        self.close_flag = None  # 控制图像显示结束的标志
        self.close_placeholder = None  # 关闭按钮区域
        self.image_placeholder = None  # 用于显示图像的区域
        self.image_placeholder_res = None  # 图像显示区域
        self.table_placeholder = None  # 表格显示区域
        self.log_table_placeholder = None  # 完整结果表格显示区域
        self.selectbox_placeholder = None  # 下拉框显示区域
        self.selectbox_target = None  # 下拉框选中项
        self.progress_bar = None  # 用于显示的进度条

        # 初始化日志数据保存路径
        self.saved_log_data = abs_path("tempDir/log_table_data.csv", path_type="current")

        # 如果在 session state 中不存在logTable，创建一个新的LogTable实例
        if 'logTable' not in st.session_state:
            st.session_state['logTable'] = LogTable(self.saved_log_data)

        # 获取或更新可用摄像头列表
        if 'available_cameras' not in st.session_state:
            st.session_state['available_cameras'] = get_camera_names()
        self.available_cameras = st.session_state['available_cameras']

        # 初始化或获取识别结果的表格
        self.logTable = st.session_state['logTable']

        # 加载或创建模型实例
        if 'model' not in st.session_state:
            st.session_state['model'] = YOLOv8v5Detector()  # 创建YOLOv8Detector模型实例

        self.model = st.session_state['model']
        # 加载训练的模型权重
        self.model.load_model(model_path=abs_path("weights/yolov8m_best.pt", path_type="current"))
        # 为模型中的类别重新分配颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]

        # 初始化邮箱登录状态
        if 'logged_in_email' in st.session_state:
            self.is_logged_in = True
            self.logged_in_email = st.session_state['logged_in_email']
            self.logged_in_password = st.session_state['logged_in_password']
        else:
            self.is_logged_in = False
            self.logged_in_email = None
            self.logged_in_password = None

        if 'target_email' in st.session_state:
            self.target_email = st.session_state['target_email']
        else:
            self.target_email = None

        # 初始化告警间隔时间
        self.alert_interval = 10  # 默认告警间隔为10秒
        self.last_alert_time = None  # 上次发送告警邮件的时间

        self.setup_sidebar()  # 初始化侧边栏布局

    def setup_page(self):
        # 设置页面布局
        st.set_page_config(
            page_title=self.title,
            page_icon="🚀",
            initial_sidebar_state="expanded"
        )

    def setup_sidebar(self):
        """
        设置 Streamlit 侧边栏。

        在侧边栏中配置模型设置、摄像头选择以及识别项目设置等选项。
        """
        # 设置侧边栏的模型设置部分
        st.sidebar.header("模型设置")
        # 选择模型类型的下拉菜单，包含预设模型和自定义选项
        model_options = ["yolov8m_best", "best", "yolov8n", "yolov8s", "自定义"]
        self.model_type = st.sidebar.selectbox("选择模型类型", model_options)

        # 如果选择自定义模型文件，则提供文件上传器
        if self.model_type == "自定义":
            model_file = st.sidebar.file_uploader("选择.pt文件", type="pt")

            # 如果上传了模型文件，则保存并加载该模型
            if model_file is not None:
                self.custom_model_file = save_uploaded_file(model_file)
                self.model.load_model(model_path=self.custom_model_file)
                self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]
        else:
            # 加载预设的模型文件
            model_path = abs_path(f"weights/{self.model_type}.pt", path_type="current")
            self.model.load_model(model_path=model_path)
            # 为模型中的类别重新分配颜色
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]

        # 置信度阈值的滑动条
        self.conf_threshold = float(st.sidebar.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.25))
        # IOU阈值的滑动条
        self.iou_threshold = float(st.sidebar.slider("IOU阈值", min_value=0.0, max_value=1.0, value=0.5))

        # 设置侧边栏的摄像头配置部分
        st.sidebar.header("摄像头配置")
        # 选择摄像头的下拉菜单
        self.selected_camera = st.sidebar.selectbox("选择摄像头", self.available_cameras)

        # 设置侧边栏的识别项目设置部分
        st.sidebar.header("识别项目设置")
        # 选择文件类型的下拉菜单
        self.file_type = st.sidebar.selectbox("选择文件类型", ["图片文件", "视频文件"], index=1)
        # 根据所选的文件类型，提供对应的文件上传器
        if self.file_type == "图片文件":
            self.uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
        elif self.file_type == "视频文件":
            self.uploaded_video = st.sidebar.file_uploader("上传视频文件", type=["mp4"])

        # 提供相关提示信息，根据所选摄像头和文件类型的不同情况
        if self.selected_camera == "未启用摄像头":
            if self.file_type == "图片文件":
                st.sidebar.write("请选择图片并点击'开始运行'按钮，进行图片检测！")
            if self.file_type == "视频文件":
                st.sidebar.write("请选择视频并点击'开始运行'按钮，进行视频检测！")
        else:
            st.sidebar.write("请点击'开始运行'按钮，启动摄像头检测！")

        # 设置侧边栏的邮箱配置部分
        st.sidebar.header("邮箱配置")
        # 输入QQ邮箱和密码
        self.logged_in_email = st.sidebar.text_input("输入QQ邮箱", value=self.logged_in_email or "")
        self.logged_in_password = st.sidebar.text_input("输入QQ邮箱密码", type="password", value=self.logged_in_password or "")
        # 登录按钮
        if st.sidebar.button("登录"):
            try:
                self.is_logged_in = self.login_email()
                if self.is_logged_in:
                    st.session_state['logged_in_email'] = self.logged_in_email
                    st.session_state['logged_in_password'] = self.logged_in_password
                    st.sidebar.success("登录成功")
                else:
                    st.sidebar.error("登录失败，请检查邮箱和密码")
            except Exception as e:
                st.sidebar.error(f"登录失败: {str(e)}")
        # 显示当前登录的邮箱
        if 'logged_in_email' in st.session_state and st.session_state['logged_in_email']:
            st.sidebar.write(f"当前登录邮箱: {st.session_state['logged_in_email']}")
            # 登出按钮
            if st.sidebar.button("登出"):
                self.is_logged_in = False
                del st.session_state['logged_in_email']
                del st.session_state['logged_in_password']
                st.sidebar.success("已登出")
        # 输入目标邮箱
        self.target_email = st.sidebar.text_input("输入接收邮箱", value=self.target_email or "", key="target_email_input")
        if st.sidebar.button("确定", key="confirm_target_email"):
            if self.target_email:
                st.session_state['target_email'] = self.target_email
                st.sidebar.success("收件邮箱已锁定")
            else:
                st.sidebar.error("请输入有效的邮箱地址")
        # 显示当前锁定的收件邮箱
        if 'target_email' in st.session_state and st.session_state['target_email']:
            st.sidebar.write(f"当前锁定的收件邮箱: {st.session_state['target_email']}")
            # 解锁按钮
            if st.sidebar.button("解锁", key="unlock_target_email"):
                del st.session_state['target_email']
                st.sidebar.success("收件邮箱已解锁")
        # 手动告警按钮
        if st.sidebar.button("手动告警"):
            if 'logged_in_email' in st.session_state and 'target_email' in st.session_state:
                try:
                    self.send_email("手动告警")
                    st.sidebar.success("手动告警邮件已发送")
                except Exception as e:
                    st.sidebar.error(f"手动告警邮件发送失败: {str(e)}")
            else:
                st.sidebar.error("请先登录邮箱并设置目标邮箱")

        # 设置侧边栏的告警间隔配置部分
        st.sidebar.header("告警配置")
        # 告警间隔时间的滑动条，范围从0秒（立即告警）到3600秒（1小时）
        self.alert_interval = st.sidebar.slider("告警间隔（秒）", min_value=0, max_value=3600, value=30)

        # 添加声音提示的开关
        st.sidebar.header("声音提示")
        self.enable_sound = st.sidebar.checkbox("开启声音提示", value=True)

    def load_model_file(self):
        if self.custom_model_file:
            self.model.load_model(self.custom_model_file)
        else:
            pass  # 载入

    def process_camera_or_file(self):
        """
        处理摄像头或文件输入。

        根据用户选择的输入源（摄像头、图片文件或视频文件），处理并显示检测结果。
        """
        # 如果选择了摄像头输入
        if self.selected_camera != "未启用摄像头":
            self.logTable.clear_frames()  # 清除之前的帧记录
            # 创建一个结束按钮
            self.close_flag = self.close_placeholder.button(label="停止")

            # 使用 OpenCV 捕获摄像头画面
            cap = cv2.VideoCapture(int(self.selected_camera))

            # 设置总帧数为1000
            total_frames = 1000
            current_frame = 0
            self.progress_bar.progress(0)  # 初始化进度条
            while cap.isOpened() and not self.close_flag:
                ret, frame = cap.read()
                if ret:
                    # 显示画面并处理结果
                    image, detInfo, _ = self.frame_process(frame, "Camera: " + self.selected_camera)

                    # 设置新的尺寸
                    new_width = 1080
                    new_height = int(new_width * (9 / 16))
                    resized_image = cv2.resize(image, (new_width, new_height))  # 调整图像尺寸
                    resized_frame = cv2.resize(frame, (new_width, new_height))

                    # 根据显示模式显示处理后的图像或原始图像
                    if self.display_mode == "单画面显示":
                        self.image_placeholder.image(resized_image, channels="BGR", caption="摄像头画面")
                    else:
                        self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                        self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")
                    # 将帧信息添加到日志表格中
                    self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                    # 更新进度条
                    progress_percentage = int((current_frame / total_frames) * 100)
                    self.progress_bar.progress(progress_percentage)
                    current_frame = (current_frame + 1) % total_frames  # 重置进度条
                else:
                    st.error("无法获取图像。")
                    break
                # time.sleep(0.01)  # 控制帧率

            # 保存结果到CSV并更新日志表格
            if self.close_flag:
                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()

            self.logTable.save_to_csv()
            self.logTable.update_table(self.log_table_placeholder)
            cap.release()
        else:
            # 如果上传了图片文件
            if self.uploaded_file is not None:
                self.logTable.clear_frames()
                self.progress_bar.progress(0)
                # 显示上传的图片
                source_img = self.uploaded_file.read()
                file_bytes = np.asarray(bytearray(source_img), dtype=np.uint8)
                image_ini = cv2.imdecode(file_bytes, 1)

                image, detInfo, select_info = self.frame_process(image_ini, self.uploaded_file.name)

                # self.selectbox_placeholder = st.empty()
                self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", select_info, key="22113")

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)  # 更新所有结果记录的表格

                # 设置新的尺寸
                new_width = 1080
                new_height = int(new_width * (9 / 16))
                # 调整图像尺寸
                resized_image = cv2.resize(image, (new_width, new_height))
                resized_frame = cv2.resize(image_ini, (new_width, new_height))
                if self.display_mode == "单画面显示":
                    self.image_placeholder.image(resized_image, channels="BGR", caption="图片显示")
                else:
                    self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

                self.logTable.add_frames(image, detInfo, cv2.resize(image_ini, (640, 640)))
                self.progress_bar.progress(100)

            # 如果上传了视频文件
            elif self.uploaded_video is not None:
                # 处理上传的视频
                self.logTable.clear_frames()
                self.close_flag = self.close_placeholder.button(label="停止")

                video_file = self.uploaded_video
                tfile = tempfile.NamedTemporaryFile()
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)

                # 获取视频总帧数和帧率
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 计算视频总长度（秒）
                total_length = total_frames / fps if fps > 0 else 0

                # 创建进度条
                self.progress_bar.progress(0)

                current_frame = 0
                while cap.isOpened() and not self.close_flag:
                    ret, frame = cap.read()
                    if ret:
                        image, detInfo, _ = self.frame_process(frame, self.uploaded_video.name)

                        # 设置新的尺寸
                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        # 调整图像尺寸
                        resized_image = cv2.resize(image, (new_width, new_height))
                        resized_frame = cv2.resize(frame, (new_width, new_height))
                        if self.display_mode == "单画面显示":
                            self.image_placeholder.image(resized_image, channels="BGR", caption="视频画面")
                        else:
                            self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                            self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

                        self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                        # 更新进度条
                        if total_length > 0:
                            progress_percentage = int(((current_frame + 1) / total_frames) * 100)
                            self.progress_bar.progress(progress_percentage)

                        current_frame += 1
                    else:
                        break
                if self.close_flag:
                    self.logTable.save_to_csv()
                    self.logTable.update_table(self.log_table_placeholder)
                    cap.release()

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()

            else:
                st.warning("请选择摄像头或上传文件。")

    def toggle_comboBox(self, frame_id):
        """
        处理并显示指定帧的检测结果。

        Args:
            frame_id (int): 指定要显示检测结果的帧ID。

        根据用户选择的帧ID，显示该帧的检测结果和图像。
        """
        # 确保已经保存了检测结果
        if len(self.logTable.saved_results) > 0:
            frame = self.logTable.saved_images_ini[-1]  # 获取最近一帧的图像
            image = frame  # 将其设为当前图像

            # 遍历所有保存的检测结果
            for i, detInfo in enumerate(self.logTable.saved_results):
                if frame_id != -1:
                    # 如果指定了帧ID，只处理该帧的结果
                    if frame_id != i:
                        continue

                if len(detInfo) > 0:
                    name, bbox, conf, use_time, cls_id = detInfo  # 获取检测信息
                    label = '%s %.0f%%' % (name, conf * 100)  # 构造标签文本

                    disp_res = ResultLogger()  # 创建结果记录器
                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(round(use_time, 2)))  # 合并结果
                    self.table_placeholder.table(res)  # 在表格中显示结果

                    # 如果有保存的初始图像
                    if len(self.logTable.saved_images_ini) > 0:
                        if len(self.colors) < cls_id:
                            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(cls_id+1)]
                        image = drawRectBox(image, bbox, alpha=0.2, addText=label,
                                            color=self.colors[cls_id])  # 绘制检测框和标签

            # 设置新的尺寸并调整图像尺寸
            new_width = 1080
            new_height = int(new_width * (9 / 16))
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # 根据显示模式显示处理后的图像或原始图像
            if self.display_mode == "单画面显示":
                self.image_placeholder.image(resized_image, channels="BGR", caption="识别画面")
            else:
                self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

    def frame_process(self, image, file_name):
        """
        处理并预测单个图像帧的内容。

        Args:
            image (numpy.ndarray): 输入的图像。
            file_name (str): 处理的文件名。

        Returns:
            tuple: 处理后的图像，检测信息，选择信息列表。

        对输入图像进行预处理，使用模型进行预测，并处理预测结果。
        """
        image = cv2.resize(image, (640, 640))  # 调整图像大小以适应模型
        pre_img = self.model.preprocess(image)  # 对图像进行预处理

        # 更新模型参数
        params = {'conf': self.conf_threshold, 'iou': self.iou_threshold}
        self.model.set_param(params)

        t1 = time.time()
        pred = self.model.predict(pre_img)  # 使用模型进行预测
        t2 = time.time()
        use_time = t2 - t1  # 计算单张图片推理时间

        det = pred[0]  # 获取预测结果

        # 初始化检测信息和选择信息列表
        detInfo = []
        select_info = ["全部目标"]

        # 如果有有效的检测结果
        if det is not None and len(det):
            det_info = self.model.postprocess(pred)  # 后处理预测结果
            if len(det_info):
                disp_res = ResultLogger()
                res = None
                cnt = 0

                # 遍历检测到的对象
                for info in det_info:
                    name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
                    label = '%s %.0f%%' % (name, conf * 100)

                    # 如果当前目标被过滤，跳过检测
                    if self.selectbox_target != "全部目标" and name == self.selectbox_target.split("-")[0]:
                        continue

                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(round(use_time, 2)))

                    # 绘制检测框和标签
                    image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=self.colors[cls_id])
                    # 添加日志条目
                    self.logTable.add_log_entry(file_name, name, bbox, conf, use_time)
                    # 记录检测信息
                    detInfo.append([name, bbox, conf, use_time, cls_id])
                    # 添加到选择信息列表
                    select_info.append(name + "-" + str(cnt))
                    cnt += 1

                    # 实时更新表格显示
                    self.table_placeholder.table(res)

                    # 如果检测到异常行为且置信度大于0.75，处理提醒
                    if name in ["吸烟","奔跑","翻墙","打架","摔倒","玩手机","举手"] and conf > 0.75:
                        current_time = time.time()
                        # 检查是否满足告警间隔条件
                        if self.last_alert_time is None or (current_time - self.last_alert_time) >= self.alert_interval:
                            # 如果已登录邮箱，发送邮件
                            if self.is_logged_in:
                                threading.Thread(target=self.send_email_async, args=(name,)).start()
                                st.toast(f"已发送邮件提醒: {name}", icon="🚨")
                            else:
                                # 未登录邮箱，只显示检测到异常行为的提示
                                st.toast(f"检测到异常行为: {name}", icon="🚨")
                            # 播放声音提示
                            if self.enable_sound:
                                threading.Thread(target=playsound, args=("alert.wav",)).start()
                            self.last_alert_time = current_time

        return image, detInfo, select_info

    def setupMainWindow(self):
        """
        运行行人跌倒检测系统。

        构建并显示行人跌倒检测系统的主界面，包括图像显示、控制选项、结果展示等。
        """
        st.title(self.title)  # 显示系统标题
        st.write("--------")
        st.write("计科B21-4 2123819 程飞扬 导师：董辛酉")
        st.write("--------")  # 插入一条分割线

        # 创建列布局
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 1])

        # 在第一列设置显示模式的选择
        with col1:
            self.display_mode = st.radio("显示模式", ["单画面显示", "双画面显示"])

        # 根据显示模式创建用于显示视频画面的空容器
        if self.display_mode == "单画面显示":
            self.image_placeholder = st.empty()
            if not self.logTable.saved_images_ini:
                self.image_placeholder.image(load_default_image(), caption="原始画面")
        else:  # "双画面显示"
            self.image_placeholder = st.empty()
            self.image_placeholder_res = st.empty()
            if not self.logTable.saved_images_ini:
                self.image_placeholder.image(load_default_image(), caption="原始画面")
                self.image_placeholder_res.image(load_default_image(), caption="识别画面")

        # 显示用的进度条
        self.progress_bar = st.progress(0)

        # 创建一个空的结果表格
        res = concat_results("None", "[0, 0, 0, 0]", "0.00", "0.00s")
        self.table_placeholder = st.empty()
        self.table_placeholder.table(res)

        # 创建一个导出结果的按钮
        st.write("---------------------")
        if st.button("导出结果"):
            self.logTable.save_to_csv()
            res = self.logTable.save_frames_file()
            st.write("🚀识别结果文件已经保存：" + self.saved_log_data)
            if res:
                st.write(f"🚀结果的视频/图片文件已经保存：{res}")
            self.logTable.clear_data()

        # 显示所有结果记录的空白表格
        self.log_table_placeholder = st.empty()
        self.logTable.update_table(self.log_table_placeholder)

        # 在第五列设置一个空的停止按钮占位符
        with col5:
            st.write("")
            self.close_placeholder = st.empty()

        # 在第二列处理目标过滤
        with col2:
            self.selectbox_placeholder = st.empty()
            detected_targets = ["全部目标","吸烟","奔跑","翻墙","打架","摔倒","玩手机","举手"]  # 初始化目标列表

            # 遍历并显示检测结果
            for i, info in enumerate(self.logTable.saved_results):
                name, bbox, conf, use_time, cls_id = info
                detected_targets.append(name + "-" + str(i))
            self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", detected_targets)

            # 处理目标过滤的选择
            for i, info in enumerate(self.logTable.saved_results):
                name, bbox, conf, use_time, cls_id = info
                if self.selectbox_target == name + "-" + str(i):
                    self.toggle_comboBox(i)
                elif self.selectbox_target == "全部目标":
                    self.toggle_comboBox(-1)

        # 在第四列设置一个开始运行的按钮
        with col4:
            st.write("")
            run_button = st.button("开始运行")

            if run_button:
                self.process_camera_or_file()  # 运行摄像头或文件处理
            else:
                # 如果没有保存的图像，则显示默认图像
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="原始画面")
                    if self.display_mode == "双画面显示":
                        self.image_placeholder_res.image(load_default_image(), caption="识别画面")

    def login_email(self):
        """
        登录QQ邮箱。
        """
        try:
            # 增加超时设置和重试机制
            smtp = smtplib.SMTP_SSL('smtp.qq.com', 465, timeout=10)
            smtp.ehlo()  # 发送EHLO命令
            smtp.login(self.logged_in_email, self.logged_in_password)
            smtp.quit()
            return True
        except smtplib.SMTPException as e:
            # 如果是连接问题，尝试重试一次
            try:
                smtp = smtplib.SMTP_SSL('smtp.qq.com', 465, timeout=10)
                smtp.ehlo()
                smtp.login(self.logged_in_email, self.logged_in_password)
                smtp.quit()
                return True
            except Exception as retry_e:
                raise retry_e
        except Exception as e:
            raise e

    def send_email_async(self, behavior_type):
        """
        异步发送邮件提醒。
        """
        if not self.is_logged_in or not self.target_email:
            st.toast("未登录邮箱或未设置目标邮箱", icon="❌")  # 使用st.toast显示提醒
            return

        try:
            # 邮件内容
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            mail_msg = f"检测时间: {current_time}\n异常行为类型: {behavior_type}"
            message = MIMEText(mail_msg, 'plain', 'utf-8')
            # 确保From头信息符合RFC标准
            from_name = "异常行为检测系统"  # 昵称
            from_email = self.logged_in_email  # 邮箱地址
            # 对昵称进行Base64编码
            from_name_encoded = Header(from_name, 'utf-8').encode()
            message['From'] = f"{from_name_encoded} <{from_email}>"
            message['To'] = Header(self.target_email, 'utf-8')
            message['Subject'] = Header('异常行为检测提醒', 'utf-8')

            # 发送邮件，增加超时设置
            smtp = smtplib.SMTP_SSL('smtp.qq.com', 465, timeout=10)
            smtp.ehlo()  # 发送EHLO命令
            smtp.login(self.logged_in_email, self.logged_in_password)
            smtp.sendmail(self.logged_in_email, self.target_email, message.as_string())
            smtp.quit()
            st.toast("邮件发送成功", icon="✅")  # 使用st.toast显示提醒
        except smtplib.SMTPException as e:
            # 如果是连接问题，尝试重试一次
            try:
                smtp = smtplib.SMTP_SSL('smtp.qq.com', 465, timeout=10)
                smtp.ehlo()
                smtp.login(self.logged_in_email, self.logged_in_password)
                smtp.sendmail(self.logged_in_email, self.target_email, message.as_string())
                smtp.quit()
                st.toast("邮件发送成功", icon="✅")  # 使用st.toast显示提醒
            except Exception as retry_e:
                st.toast(f"邮件发送失败: {str(retry_e)}", icon="❌")  # 使用st.toast显示提醒
        except Exception as e:
            st.toast(f"邮件发送过程中发生未知错误: {str(e)}", icon="❌")  # 使用st.toast显示提醒

# 实例化并运行应用
if __name__ == "__main__":
    app = Detection_UI()
    app.setupMainWindow()