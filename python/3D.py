import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QSlider, QGroupBox, QSplitter, QTextEdit, QMessageBox,
                             QStatusBar, QCheckBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QFont, QColor, QImage, QOpenGLTexture

import OpenGL.GL as gl
import trimesh
import json
import base64
from io import BytesIO
from PIL import Image


class TextureManager:
    """纹理管理器，处理GLB/GLTF纹理加载"""
    
    def __init__(self):
        self.textures = {}  # 存储OpenGL纹理ID
        self.images = {}    # 存储PIL图像
        
    def load_texture_from_glb(self, gltf_data, image_index):
        """从GLB数据加载纹理"""
        try:
            if image_index in self.textures:
                return self.textures[image_index]
                
            if 'images' not in gltf_data or image_index >= len(gltf_data['images']):
                return None
                
            image_info = gltf_data['images'][image_index]
            image_data = None
            
            # 从bufferView加载
            if 'bufferView' in image_info:
                buffer_view_idx = image_info['bufferView']
                buffer_view = gltf_data['bufferViews'][buffer_view_idx]
                buffer_idx = buffer_view['buffer']
                
                # 获取buffer数据
                if 'buffers' in gltf_data:
                    buffer_data = self.get_buffer_data(gltf_data, buffer_idx)
                    if buffer_data:
                        offset = buffer_view.get('byteOffset', 0)
                        length = buffer_view['byteLength']
                        image_data = buffer_data[offset:offset+length]
                        
            # 从URI加载（外部文件）
            elif 'uri' in image_info:
                uri = image_info['uri']
                if uri.startswith('data:'):
                    # Data URI
                    header, data = uri.split(',', 1)
                    image_data = base64.b64decode(data)
                else:
                    # 外部文件，需要单独加载
                    pass
                    
            if image_data:
                # 使用PIL打开图像
                image = Image.open(BytesIO(image_data))
                # 转换为RGBA
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                    
                self.images[image_index] = image
                
                # 创建OpenGL纹理
                texture_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
                
                # 上传纹理数据
                img_data = np.array(image, dtype=np.uint8)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 
                               image.width, image.height, 0,
                               gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
                
                # 设置纹理参数
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                
                self.textures[image_index] = texture_id
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                return texture_id
                
        except Exception as e:
            print(f"加载纹理失败: {e}")
            
        return None
        
    def get_buffer_data(self, gltf_data, buffer_index):
        """获取buffer数据"""
        if 'buffers' not in gltf_data or buffer_index >= len(gltf_data['buffers']):
            return None
            
        buffer_info = gltf_data['buffers'][buffer_index]
        
        # 如果已经加载了二进制数据
        if hasattr(self, 'binary_data') and self.binary_data:
            return self.binary_data
            
        # 从URI加载
        if 'uri' in buffer_info:
            uri = buffer_info['uri']
            if uri.startswith('data:application/octet-stream;base64,'):
                data = uri.split(',', 1)[1]
                return base64.b64decode(data)
                
        return None
        
    def cleanup(self):
        """清理纹理资源"""
        for tex_id in self.textures.values():
            gl.glDeleteTextures(1, [tex_id])
        self.textures.clear()
        self.images.clear()


class ModernGLWidget(QOpenGLWidget):
    """OpenGL渲染窗口，支持3D模型显示、纹理和交互"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # 相机参数
        self.camera_distance = 5.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_target = np.array([0.0, 0.0, 0.0])
        
        # 模型变换
        self.model_matrix = np.eye(4, dtype=np.float32)
        self.model_scale = 1.0
        self.model_offset = np.array([0.0, 0.0, 0.0])
        
        # 鼠标交互
        self.last_mouse_pos = QPoint()
        self.is_rotating = False
        self.is_panning = False
        
        # 模型数据
        self.current_mesh = None
        self.gltf_data = None  # 原始GLTF数据，用于纹理
        self.binary_data = None  # GLB二进制数据
        
        self.vertices = None
        self.faces = None
        self.normals = None
        self.uvs = None  # 纹理坐标
        self.vertex_colors = None
        self.bounding_box = None
        
        # 材质和纹理
        self.has_texture = False
        self.texture_id = None
        self.base_color = [0.8, 0.8, 0.8, 1.0]  # 默认灰色
        self.metallic = 0.0
        self.roughness = 0.5
        
        # 渲染参数
        self.bg_color = QColor(35, 35, 45)
        self.wireframe_mode = False
        self.show_axes = True
        self.show_bbox = False
        self.use_texture = True
        self.point_size = 2.0
        
        # 动画
        self.auto_rotate = False
        self.rotation_speed = 0.5
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_rotation)
        
        # 纹理管理器
        self.texture_manager = TextureManager()
        
    def initializeGL(self):
        """初始化OpenGL环境"""
        gl.glClearColor(self.bg_color.redF(), self.bg_color.greenF(), 
                       self.bg_color.blueF(), 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_NORMALIZE)
        
        # 启用纹理
        gl.glEnable(gl.GL_TEXTURE_2D)
        
        # 光照
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [10.0, 10.0, 10.0, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # 材质
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        
        # 混合
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
    def resizeGL(self, width, height):
        """窗口大小改变时调整视口"""
        if height == 0:
            height = 1
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / height
        glu.gluPerspective(45.0, aspect, 0.01, 1000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
    def paintGL(self):
        """渲染场景"""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        
        self.setup_camera()
        self.draw_grid()
        
        if self.show_axes:
            self.draw_axes()
            
        if self.vertices is not None and len(self.vertices) > 0:
            self.draw_mesh()
            
        if self.show_bbox and self.bounding_box is not None:
            self.draw_bounding_box()
            
    def setup_camera(self):
        """设置相机"""
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        elevation_rad = max(-89.0, min(89.0, elevation_rad))
        
        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.sin(elevation_rad)
        z = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        
        camera_pos = self.camera_target + np.array([x, y, z])
        
        glu.gluLookAt(
            float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2]),
            float(self.camera_target[0]), float(self.camera_target[1]), float(self.camera_target[2]),
            0.0, 1.0, 0.0
        )
        
    def draw_grid(self):
        """绘制网格地面"""
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)
        
        if self.bounding_box is not None:
            size = np.max(self.bounding_box[1] - self.bounding_box[0])
            grid_size = int(size * 2) + 2
            grid_step = max(0.1, size / 10)
        else:
            grid_size = 10
            grid_step = 1.0
            
        gl.glColor3f(0.25, 0.25, 0.3)
        gl.glBegin(gl.GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            gl.glVertex3f(i * grid_step, 0.0, -grid_size * grid_step)
            gl.glVertex3f(i * grid_step, 0.0, grid_size * grid_step)
            gl.glVertex3f(-grid_size * grid_step, 0.0, i * grid_step)
            gl.glVertex3f(grid_size * grid_step, 0.0, i * grid_step)
        gl.glEnd()
        
        gl.glEnable(gl.GL_LIGHTING)
        
    def draw_axes(self):
        """绘制坐标轴"""
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(3.0)
        
        axis_length = 1.5
        if self.bounding_box is not None:
            size = np.max(self.bounding_box[1] - self.bounding_box[0])
            axis_length = size * 0.5
        
        # X轴 - 红色
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1.0, 0.2, 0.2)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(axis_length, 0.0, 0.0)
        gl.glEnd()
        
        # Y轴 - 绿色
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.2, 1.0, 0.2)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, axis_length, 0.0)
        gl.glEnd()
        
        # Z轴 - 蓝色
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.2, 0.4, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, axis_length)
        gl.glEnd()
        
        gl.glEnable(gl.GL_LIGHTING)
        
    def draw_bounding_box(self):
        """绘制包围盒"""
        if self.bounding_box is None:
            return
            
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(2.0)
        gl.glColor3f(1.0, 0.8, 0.2)
        
        min_pt, max_pt = self.bounding_box
        vertices = [
            [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
        ]
        edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
        
        gl.glBegin(gl.GL_LINES)
        for edge in edges:
            gl.glVertex3fv(vertices[edge[0]])
            gl.glVertex3fv(vertices[edge[1]])
        gl.glEnd()
        
        gl.glEnable(gl.GL_LIGHTING)
        
    def draw_mesh(self):
        """绘制3D模型（支持纹理）"""
        if self.vertices is None or len(self.vertices) == 0:
            return
            
        gl.glPushMatrix()
        
        # 应用模型变换
        gl.glTranslatef(
            float(self.model_offset[0]),
            float(self.model_offset[1]),
            float(self.model_offset[2])
        )
        gl.glScalef(self.model_scale, self.model_scale, self.model_scale)
        
        # 设置材质
        if not self.wireframe_mode:
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, self.base_color)
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, (1.0 - self.roughness) * 128.0)
        
        # 绑定纹理
        if self.use_texture and self.has_texture and self.texture_id is not None and not self.wireframe_mode:
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glColor3f(1.0, 1.0, 1.0)  使用纹理时设置白色
        else:
            gl.glDisable(gl.GL_TEXTURE_2D)
            
        if self.wireframe_mode:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glColor3f(0.9, 0.9, 0.9)
            gl.glLineWidth(1.0)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glEnable(gl.GL_LIGHTING)
            
        # 绘制
        if self.faces is not None and len(self.faces) > 0:
            self.draw_textured_triangles()
        else:
            self.draw_point_cloud()
            
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glPopMatrix()
        
    def draw_textured_triangles(self):
        """绘制带纹理的三角形"""
        gl.glBegin(gl.GL_TRIANGLES)
        
        for face in self.faces:
            for i, vertex_idx in enumerate(face):
                # 设置纹理坐标
                if self.uvs is not None and vertex_idx < len(self.uvs):
                    uv = self.uvs[vertex_idx]
                    gl.glTexCoord2f(float(uv[0]), float(uv[1]))
                
                # 设置法线
                if self.normals is not None and vertex_idx < len(self.normals):
                    gl.glNormal3fv(self.normals[vertex_idx])
                else:
                    # 计算面法线
                    if len(face) >= 3:
                        v0 = self.vertices[face[0]]
                        v1 = self.vertices[face[1]]
                        v2 = self.vertices[face[2]]
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = np.cross(edge1, edge2)
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            normal = normal / norm
                            gl.glNormal3fv(normal)
                
                # 设置颜色（如果没有纹理）
                if not self.has_texture or self.wireframe_mode:
                    if self.vertex_colors is not None and vertex_idx < len(self.vertex_colors):
                        color = self.vertex_colors[vertex_idx]
                        gl.glColor3fv(color[:3])
                    else:
                        gl.glColor3f(self.base_color[0], self.base_color[1], self.base_color[2])
                
                # 设置顶点
                gl.glVertex3fv(self.vertices[vertex_idx])
                
        gl.glEnd()
        
    def draw_point_cloud(self):
        """绘制点云"""
        gl.glPointSize(self.point_size)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glBegin(gl.GL_POINTS)
        
        for i, vertex in enumerate(self.vertices):
            if self.vertex_colors is not None and i < len(self.vertex_colors):
                color = self.vertex_colors[i]
                gl.glColor3fv(color[:3])
            else:
                gl.glColor3f(self.base_color[0], self.base_color[1], self.base_color[2])
            gl.glVertex3fv(vertex)
            
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)
        
    def parse_gltf_textures(self, gltf_data, binary_data=None):
        """解析GLTF纹理数据"""
        self.texture_manager.binary_data = binary_data
        
        # 获取默认材质
        material = None
        if 'materials' in gltf_data and len(gltf_data['materials']) > 0:
            material = gltf_data['materials'][0]
            
        if material is None:
            return
            
        # 解析PBR材质参数
        if 'pbrMetallicRoughness' in material:
            pbr = material['pbrMetallicRoughness']
            
            # 基础颜色
            if 'baseColorFactor' in pbr:
                color = pbr['baseColorFactor']
                self.base_color = [color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0]
                
            # 基础颜色纹理
            if 'baseColorTexture' in pbr and 'textures' in gltf_data:
                texture_info = pbr['baseColorTexture']
                texture_index = texture_info.get('index', 0)
                
                if texture_index < len(gltf_data['textures']):
                    texture = gltf_data['textures'][texture_index]
                    source = texture.get('source', 0)
                    
                    # 加载纹理图像
                    tex_id = self.texture_manager.load_texture_from_glb(gltf_data, source)
                    if tex_id is not None:
                        self.texture_id = tex_id
                        self.has_texture = True
                        print(f"成功加载纹理: 索引 {source}")
                        
            # 金属度和粗糙度
            self.metallic = pbr.get('metallicFactor', 0.0)
            self.roughness = pbr.get('roughnessFactor', 0.5)
            
    def load_mesh(self, mesh, gltf_data=None, binary_data=None):
        """加载模型（支持纹理）"""
        try:
            self.current_mesh = mesh
            self.gltf_data = gltf_data
            self.binary_data = binary_data
            
            # 清理旧纹理
            if self.texture_manager:
                self.texture_manager.cleanup()
            self.texture_manager = TextureManager()
            self.texture_manager.binary_data = binary_data
            
            # 如果有GLTF数据，解析纹理
            if gltf_data is not None:
                self.parse_gltf_textures(gltf_data, binary_data)
                
            # 处理场景类型
            if isinstance(mesh, trimesh.Scene):
                print(f"加载场景，包含 {len(mesh.geometry)} 个几何体")
                combined = mesh.dump(concatenate=True)
                if isinstance(combined, list) and len(combined) > 0:
                    mesh = trimesh.util.concatenate(combined)
                elif isinstance(combined, trimesh.Trimesh):
                    mesh = combined
                else:
                    print("场景为空")
                    return
                    
            if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                print("模型没有顶点数据")
                return
                
            print(f"模型顶点数: {len(mesh.vertices)}")
            
            # 计算包围盒和变换
            self.bounding_box = mesh.bounds
            center = (self.bounding_box[0] + self.bounding_box[1]) / 2.0
            size = self.bounding_box[1] - self.bounding_box[0]
            max_size = np.max(size)
            
            self.model_offset = -center
            if max_size > 0:
                target_size = 3.0
                self.model_scale = target_size / max_size
                
            # 提取数据
            self.vertices = np.array(mesh.vertices, dtype=np.float32)
            
            if hasattr(mesh, 'faces') and mesh.faces is not None:
                self.faces = np.array(mesh.faces, dtype=np.int32)
            else:
                self.faces = None
                
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                self.normals = np.array(mesh.vertex_normals, dtype=np.float32)
            else:
                self.normals = None
                
            # 提取UV坐标
            self.uvs = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                    self.uvs = np.array(mesh.visual.uv, dtype=np.float32)
                    print(f"UV坐标数: {len(self.uvs)}")
                    
            # 提取顶点颜色
            self.vertex_colors = None
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors
                if len(colors) > 0:
                    self.vertex_colors = np.array(colors[:, :3], dtype=np.float32) / 255.0
                    
            # 设置相机
            self.camera_target = np.array([0.0, 0.0, 0.0])
            if max_size > 0:
                import math
                target_size = 3.0
                fov_rad = math.radians(45.0)
                self.camera_distance = (target_size / 2.0) / math.tan(fov_rad / 2.0) * 2.5
                
            self.camera_azimuth = 45.0
            self.camera_elevation = 30.0
            
            self.update()
            print("模型加载完成")
            
        except Exception as e:
            import traceback
            print(f"加载模型错误: {str(e)}")
            traceback.print_exc()
            raise
            
    def clear_mesh(self):
        """清除模型"""
        if self.texture_manager:
            self.texture_manager.cleanup()
            
        self.current_mesh = None
        self.gltf_data = None
        self.binary_data = None
        self.vertices = None
        self.faces = None
        self.normals = None
        self.uvs = None
        self.vertex_colors = None
        self.bounding_box = None
        self.has_texture = False
        self.texture_id = None
        self.model_scale = 1.0
        self.model_offset = np.array([0.0, 0.0, 0.0])
        self.camera_distance = 5.0
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.base_color = [0.8, 0.8, 0.8, 1.0]
        self.update()
        
    def mousePressEvent(self, event):
        """鼠标按下"""
        self.last_mouse_pos = event.pos()
        
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_rotating = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            
    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        self.is_rotating = False
        self.is_panning = False
        
    def mouseMoveEvent(self, event):
        """鼠标移动"""
        if self.last_mouse_pos.isNull():
            return
            
        delta = event.pos() - self.last_mouse_pos
        self.last_mouse_pos = event.pos()
        
        sensitivity = 0.5
        
        if self.is_rotating:
            self.camera_azimuth += delta.x() * sensitivity
            self.camera_elevation -= delta.y() * sensitivity
            self.camera_elevation = max(-89, min(89, self.camera_elevation))
            self.update()
            
        elif self.is_panning:
            azimuth_rad = np.radians(self.camera_azimuth)
            elevation_rad = np.radians(self.camera_elevation)
            
            forward = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.sin(elevation_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad)
            ])
            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / (np.linalg.norm(right) + 0.001)
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 0.001)
            
            pan_speed = self.camera_distance * 0.002
            self.camera_target += right * (-delta.x()) * pan_speed
            self.camera_target += up * delta.y() * pan_speed
            self.update()
            
    def wheelEvent(self, event):
        """滚轮缩放"""
        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.camera_distance *= zoom_factor
        self.camera_distance = max(0.1, min(1000.0, self.camera_distance))
        self.update()
        
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key.Key_Space:
            self.toggle_auto_rotate()
        elif event.key() == Qt.Key.Key_W:
            self.wireframe_mode = not self.wireframe_mode
            self.update()
        elif event.key() == Qt.Key.Key_A:
            self.show_axes = not self.show_axes
            self.update()
        elif event.key() == Qt.Key.Key_B:
            self.show_bbox = not self.show_bbox
            self.update()
        elif event.key() == Qt.Key.Key_T:
            self.use_texture = not self.use_texture
            self.update()
        elif event.key() == Qt.Key.Key_R:
            self.reset_camera()
            
    def toggle_auto_rotate(self):
        """自动旋转"""
        self.auto_rotate = not self.auto_rotate
        if self.auto_rotate:
            self.timer.start(16)
        else:
            self.timer.stop()
            
    def update_rotation(self):
        """更新旋转"""
        if self.auto_rotate:
            self.camera_azimuth += self.rotation_speed
            self.update()
            
    def reset_camera(self):
        """重置相机"""
        if self.current_mesh is not None and self.bounding_box is not None:
            self.camera_target = np.array([0.0, 0.0, 0.0])
            self.camera_azimuth = 45.0
            self.camera_elevation = 30.0
            
            size = self.bounding_box[1] - self.bounding_box[0]
            max_size = np.max(size)
            if max_size > 0:
                import math
                target_size = 3.0
                fov_rad = math.radians(45.0)
                self.camera_distance = (target_size / 2.0) / math.tan(fov_rad / 2.0) * 2.5
        else:
            self.camera_distance = 5.0
            self.camera_target = np.array([0.0, 0.0, 0.0])
            self.camera_azimuth = 45.0
            self.camera_elevation = 30.0
            
        self.update()


class GLBViewer(QMainWindow):
    """GLB文件查看器主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GLB 3D模型查看器 v3.0 - 支持纹理")
        self.setMinimumSize(1200, 800)
        
        self.setup_dark_theme()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        self.gl_widget = ModernGLWidget()
        splitter.addWidget(self.gl_widget)
        
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        splitter.setSizes([900, 300])
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 点击\"打开文件\"加载GLB/GLTF模型")
        
        self.current_file = None
        
    def setup_dark_theme(self):
        """设置深色主题"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { background-color: #1e1e2e; color: #cdd6f4; 
                     font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 10pt; }
            QPushButton { background-color: #313244; border: 1px solid #45475a; 
                         border-radius: 6px; padding: 8px 16px; color: #cdd6f4; font-weight: 500; }
            QPushButton:hover { background-color: #45475a; border-color: #585b70; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:disabled { background-color: #181825; color: #6c7086; }
            QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }
            QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 12px; 
                       padding-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #89b4fa; }
            QTextEdit { background-color: #181825; border: 1px solid #313244; 
                       border-radius: 6px; padding: 8px; color: #a6adc8; 
                       font-family: 'Consolas', 'Monaco', monospace; font-size: 9pt; }
            QLabel { color: #cdd6f4; }
            QStatusBar { background-color: #181825; color: #a6adc8; border-top: 1px solid #313244; }
            QCheckBox { spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 2px solid #45475a; }
            QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
        """)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)
        
        self.open_btn = QPushButton("📂 打开GLB/GLTF文件")
        self.open_btn.setMinimumHeight(42)
        self.open_btn.setStyleSheet("font-size: 11pt; font-weight: 600;")
        self.open_btn.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_btn)
        
        self.clear_btn = QPushButton("🗑️ 清除模型")
        self.clear_btn.setMinimumHeight(36)
        self.clear_btn.clicked.connect(self.clear_model)
        self.clear_btn.setEnabled(False)
        file_layout.addWidget(self.clear_btn)
        
        layout.addWidget(file_group)
        
        # 视图控制组
        view_group = QGroupBox("视图控制")
        view_layout = QVBoxLayout(view_group)
        view_layout.setSpacing(8)
        
        self.wireframe_btn = QPushButton("🔲 线框模式 (W)")
        self.wireframe_btn.setCheckable(True)
        self.wireframe_btn.clicked.connect(self.toggle_wireframe)
        view_layout.addWidget(self.wireframe_btn)
        
        self.texture_btn = QPushButton("🎨 使用纹理 (T)")
        self.texture_btn.setCheckable(True)
        self.texture_btn.setChecked(True)
        self.texture_btn.clicked.connect(self.toggle_texture)
        view_layout.addWidget(self.texture_btn)
        
        self.axes_btn = QPushButton("📐 显示坐标轴 (A)")
        self.axes_btn.setCheckable(True)
        self.axes_btn.setChecked(True)
        self.axes_btn.clicked.connect(self.toggle_axes)
        view_layout.addWidget(self.axes_btn)
        
        self.bbox_btn = QPushButton("📦 显示包围盒 (B)")
        self.bbox_btn.setCheckable(True)
        self.bbox_btn.clicked.connect(self.toggle_bbox)
        view_layout.addWidget(self.bbox_btn)
        
        self.rotate_btn = QPushButton("🔄 自动旋转 (Space)")
        self.rotate_btn.setCheckable(True)
        self.rotate_btn.clicked.connect(self.toggle_auto_rotate)
        view_layout.addWidget(self.rotate_btn)
        
        self.reset_btn = QPushButton("🎯 重置视角 (R)")
        self.reset_btn.clicked.connect(self.reset_view)
        view_layout.addWidget(self.reset_btn)
        
        layout.addWidget(view_group)
        
        # 模型信息组
        info_group = QGroupBox("模型信息")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("暂无模型信息...")
        self.info_text.setMaximumHeight(200)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # 操作提示组
        help_group = QGroupBox("操作提示")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setText("""🖱️ 鼠标控制：
• 左键拖拽：旋转视角
• 右键拖拽：平移视角
• 滚轮：缩放视角

⌨️ 快捷键：
• W：切换线框模式
• T：切换纹理显示
• A：切换坐标轴显示
• B：切换包围盒显示
• Space：自动旋转
• R：重置视角

💡 提示：
模型会自动居中并缩放到合适大小
支持GLB/GLTF纹理（如果嵌入在文件中）
""")
        help_text.setMaximumHeight(220)
        help_layout.addWidget(help_text)
        
        layout.addWidget(help_group)
        
        layout.addStretch()
        
        return panel
        
    def open_file(self):
        """打开文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择3D模型文件",
            "",
            "3D模型 (*.glb *.gltf *.obj *.stl *.ply);;GLB/GLTF (*.glb *.gltf);;所有文件 (*.*)"
        )
        
        if file_path:
            self.load_model(file_path)
            
    def load_model(self, file_path):
        """加载模型文件（支持纹理解析）"""
        try:
            self.status_bar.showMessage(f"正在加载: {os.path.basename(file_path)}...")
            QApplication.processEvents()
            
            # 根据扩展名选择加载方式
            ext = os.path.splitext(file_path)[1].lower()
            
            gltf_data = None
            binary_data = None
            
            if ext == '.glb':
                # 解析GLB文件结构
                gltf_data, binary_data = self.parse_glb_file(file_path)
                # 使用trimesh加载几何体
                mesh = trimesh.load(file_path, force='mesh')
            elif ext == '.gltf':
                # GLTF文件（可能有外部纹理）
                with open(file_path, 'r', encoding='utf-8') as f:
                    gltf_data = json.load(f)
                mesh = trimesh.load(file_path, force='mesh')
            else:
                # 其他格式（OBJ, STL等）
                mesh = trimesh.load(file_path, force='mesh')
                
            # 发送到OpenGL窗口（传递GLTF数据用于纹理）
            self.gl_widget.load_mesh(mesh, gltf_data, binary_data)
            
            # 更新UI
            self.current_file = file_path
            self.clear_btn.setEnabled(True)
            self.update_model_info(mesh, gltf_data)
            
            tex_status = "✓ 有纹理" if self.gl_widget.has_texture else "✗ 无纹理"
            self.status_bar.showMessage(f"✓ 已加载: {os.path.basename(file_path)} | {tex_status} | "
                                      f"缩放: {self.gl_widget.model_scale:.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"无法加载文件:\n{str(e)}")
            self.status_bar.showMessage("✗ 加载失败")
            
    def parse_glb_file(self, file_path):
        """解析GLB二进制格式，提取JSON和BIN数据"""
        with open(file_path, 'rb') as f:
            data = f.read()
            
        # GLB头部
        magic = int.from_bytes(data[0:4], 'little')
        version = int.from_bytes(data[4:8], 'little')
        total_length = int.from_bytes(data[8:12], 'little')
        
        if magic != 0x46546C67:  # 'glTF' in ASCII
            raise ValueError("不是有效的GLB文件")
            
        # 读取chunks
        offset = 12
        json_data = None
        binary_data = None
        
        while offset < total_length:
            chunk_length = int.from_bytes(data[offset:offset+4], 'little')
            chunk_type = int.from_bytes(data[offset+4:offset+8], 'little')
            chunk_data = data[offset+8:offset+8+chunk_length]
            
            if chunk_type == 0x4E4F534A:  # JSON
                json_data = json.loads(chunk_data.decode('utf-8'))
            elif chunk_type == 0x004E4942:  # BIN
                binary_data = chunk_data
                
            offset += 8 + chunk_length
            
        return json_data, binary_data
        
    def update_model_info(self, mesh, gltf_data=None):
        """更新模型信息显示"""
        info = []
        
        processed_mesh = mesh
        if isinstance(mesh, trimesh.Scene):
            info.append(f"类型: 场景 (Scene)")
            info.append(f"几何体数: {len(mesh.geometry)}")
            combined = mesh.dump(concatenate=True)
            if isinstance(combined, list) and len(combined) > 0:
                processed_mesh = trimesh.util.concatenate(combined)
            elif isinstance(combined, trimesh.Trimesh):
                processed_mesh = combined
                
        if processed_mesh is not None:
            if hasattr(processed_mesh, 'vertices'):
                info.append(f"顶点数: {len(processed_mesh.vertices):,}")
            if hasattr(processed_mesh, 'faces'):
                info.append(f"面数: {len(processed_mesh.faces):,}")
            if hasattr(processed_mesh, 'bounds'):
                bounds = processed_mesh.bounds
                size = bounds[1] - bounds[0]
                info.append(f"原始尺寸: {size[0]:.3f} × {size[1]:.3f} × {size[2]:.3f}")
                
        # 纹理信息
        if self.gl_widget.has_texture:
            info.append(f"纹理: ✓ 已加载")
            if self.gl_widget.texture_id:
                info.append(f"纹理ID: {self.gl_widget.texture_id}")
        else:
            info.append(f"纹理: ✗ 无")
            
        if self.gl_widget.uvs is not None:
            info.append(f"UV坐标: ✓ ({len(self.gl_widget.uvs)}个)")
        else:
            info.append(f"UV坐标: ✗ 无")
            
        info.append(f"缩放比例: {self.gl_widget.model_scale:.4f}")
        info.append(f"基础颜色: [{self.gl_widget.base_color[0]:.2f}, "
                   f"{self.gl_widget.base_color[1]:.2f}, "
                   f"{self.gl_widget.base_color[2]:.2f}]")
            
        self.info_text.setText("\n".join(info))
        
    def clear_model(self):
        """清除当前模型"""
        self.gl_widget.clear_mesh()
        self.current_file = None
        self.clear_btn.setEnabled(False)
        self.info_text.clear()
        self.wireframe_btn.setChecked(False)
        self.rotate_btn.setChecked(False)
        self.bbox_btn.setChecked(False)
        self.texture_btn.setChecked(True)
        self.axes_btn.setChecked(True)
        self.status_bar.showMessage("模型已清除")
        
    def toggle_wireframe(self):
        """切换线框模式"""
        self.gl_widget.wireframe_mode = self.wireframe_btn.isChecked()
        self.gl_widget.update()
        
    def toggle_texture(self):
        """切换纹理显示"""
        self.gl_widget.use_texture = self.texture_btn.isChecked()
        self.gl_widget.update()
        
    def toggle_axes(self):
        """切换坐标轴显示"""
        self.gl_widget.show_axes = self.axes_btn.isChecked()
        self.gl_widget.update()
        
    def toggle_bbox(self):
        """切换包围盒显示"""
        self.gl_widget.show_bbox = self.bbox_btn.isChecked()
        self.gl_widget.update()
        
    def toggle_auto_rotate(self):
        """切换自动旋转"""
        self.gl_widget.toggle_auto_rotate()
        
    def reset_view(self):
        """重置视角"""
        self.gl_widget.reset_camera()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Segoe UI", 10)
    if not QFont(font).exactMatch():
        font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = GLBViewer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
