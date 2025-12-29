# TextureProjector (Gemini AI Blender Addon) 🎨🚀

**TextureProjector** 是一款基于 Google Gemini AI 的新时代贴图工具。抛弃繁琐的 Tag 堆砌，直接通过自然语言描述即可为 3D 模型生成高质量贴图。

## ✨ 核心功能

- **自然语言驱动**: 像和人说话一样描述需求，AI 自动理解光影、材质与艺术风格。
- **智能视角还原**: 自动记录每一张历史贴图的相机状态，支持一键“瞬移”回视角进行重投影。
- **高保真自动化烘焙**: 深度集成 Blender 原生烘焙引擎，自动优化色彩空间（Standard Transform）与采样，杜绝抖动生成的噪点。
- **多样化生成引导**: 支持深度图 (Depth)、视角颜色 (Color) 或自定义图片 (Image) 作为生成源。

## 🚀 快速上手

1. **安装**: 在 Blender 插件设置中安装 ZIP。
2. **配置**: 在侧边栏 `Gemini` 面板填入 [API Key](https://aistudio.google.com/)。
3. **投影**: 
    - 选中模型并进入**编辑模式**，选中目标面。
    - 在面板中输入自然语言描述（例：“老旧且生锈的重型工业金属板”）。
    - 点击 **AI Texture Projection** 等待生成与投射。
4. **烘焙**: 
    - 确保面板中的 **Bake Result to Original UVs** 已勾选，投射完成后会自动烘焙到模型 UV 贴图上。
5. **历史还原**:
    - 在 `Projection Gallery` 中点击图片旁的齿轮图标，选择 **Use as Projection Source**，视角会自动还原到当初渲染时的位置进行完美重投影。

---

## 💡 提示词建议 (Natural Language)

无需复杂的标签，尝试更直观的句子：
- "精致的 PVC 手办质感，极其平滑的塑料表面，配合明亮的工作室灯光。"
- "干净的日系动漫平涂风格，线条洗练，色彩鲜亮，没有厚重的阴影。"
- "一个带有磨损划痕的赛博朋克金属墙壁，边缘露出了蓝色的生锈层。"

---
*Powered by Advanced Agentic Coding.*
