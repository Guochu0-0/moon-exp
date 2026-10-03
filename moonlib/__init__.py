"""baseline 推理与微调训练共用的部分：原始影像到网络输入的映射（inputs）、统一的仿射 RANSAC（ransac）。
只依赖 numpy + cv2，兼容 py3.8，matcher 环境与工作台环境都能 import。"""
