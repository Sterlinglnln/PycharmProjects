## 头部点云重构流程
# ros2 驱动 intel realsense d435i
bash: ros2 launch realsense2_camera rs_launch.py   enable_color:=true   enable_depth:=true
      depth_module.depth_profile:='848,480,30'   rgb_camera.color_profile:='848,480,30'    
      align_depth.enable:=true   pointcloud.enable:=true   pointcloud.ordered_pc:=true   
      spatial_filter.enable:=true   temporal_filter.enable:=true   hole_filling_filter.enable:=true

# ros2 bag record 录制数据包
bash: ros2 bag record \
  /camera/camera/depth/color/points \
  /camera/camera/color/image_raw \
  /camera/camera/color/camera_info \
  /camera/camera/depth/image_rect_raw \
  /camera/camera/depth/camera_info \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/aligned_depth_to_color/camera_info


# 从 mcap 包中导出所有点云帧（ply格式）存储在 ply_output 文件夹
export_ply_from_bag.py

# 点云预处理、抽帧、降采样
preprocess_pointclouds.py

# 点云 ICP 配准和点云融合导出为 merged_head.ply
register_and_merge.py

# Poisson 网格重建（mesh）输出为 head_mesh.ply
reconstruct_mesh.py

## 总结与反思
# 重建结果并不理想原因
1. 人旋转太快（相邻帧差异太大，导致 ICP 无法找对应点，错误对齐 → 出现多个头
2. 人身体、手、头发都有运动，人体不是刚性物体，ICP 无法处理“人动”的点云 → 轨迹崩坏
3. 背景、衣服、手臂大量干扰，ICP 会对错对象 → 造成漂移
4. realSense 原始点云噪声大，特别是头发区域，深度不稳定 → 重建模糊
总结： 调整采集方式

## TODO
# Step 1：重新录制点云
正确录制方法（必须照这个做）：
✔ 人保持完全静止
手不要动，身体不要动，头微微抬起固定住
✔ 相机绕着人转，而不是人转
这是专业头模扫描仪的标准做法。
✔ 转动要非常慢（1 圈 12–15 秒）
保证相邻帧差异小（2°以内）。
✔ 镜头只拍头部，不要拍全身
减少 ICP 干扰。
✔ 使用 RealSense viewer 或 ros2，相机距离保持 0.4–0.7m

# Step 2：对点云做自动裁剪（保留头部）
深度阈值切割
背景平面去除
基于颜色或区域提取只保留头部
这样 ICP 对齐更稳定。

# Step 3：关键帧配准（替代逐帧 ICP）
✔ 每隔几帧抽关键帧
✔ 所有关键帧都对齐到第一帧（用 RANSAC + ICP）
✔ 不累积误差
✔ 头部不会再出现两个影子

# Step 4：TSDF 融合
TSDF（Truncated Signed Distance Function）可以：
平滑噪声
填补洞
输出干净体积模型
比你当前的点云合并要强得多。

# Step 5：Poisson 网格重建（最终生成头部模型）
你今天跑的是 Poisson，但输入点云质量太低。
明天的 TSDF 输出会让 Poisson 效果大幅改善。
# date: Mon Dec 8

## date: Wed Dec 10
# 今日总结
# 成功点
头部轮廓已经比较完整
整体形状（头发、额头、脸侧）能看出来，说明 ICP + 多帧融合有效。

深度连续、表面融合比较平滑
没有严重裂缝，说明 TSDF / Poisson 基本成功。

颜色映射基本正确
说明点云 RGB 颜色解析是对的。
# 主要问题
1) 脸部细节丢失（严重模糊）
表现： 眼睛、鼻子、嘴基本消失 整张脸像被糊成一团
原因：
RealSense 深度在 正面有光照、表面平滑时精度最差 你的人是移动扫描 → 运动模糊 + ICP 不稳定 对齐时有累积漂移（drift）
用 ICP 逐帧对齐 + 关键帧策略 + TSDF 融合
2) 头发孔洞明显（白块、缺边）
原因： 深色头发吸光，IR 深度无法返回信号 RealSense 对黑色物体原生就弱
录制时补光（非常关键）
使用 infrared mode 或者在 ROS2 中启用 emitter on
3) 脸部位置偏移（ghosting 重影）
从图片看出： 左右脸线条不一致 是融合了多个位置不一致的脸
原因： 你采用的是：人动 → 相机不动 每帧之间旋转量较大 ICP 失败 → 错误对齐 → 多帧叠加 ghosting
必须用 参考帧配准（register_to_ref）+ 双向 ICP 检查失败帧