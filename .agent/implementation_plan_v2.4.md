# Logic Segmenter v2.4 Implementation Plan

## Overview
基于最新研究资料，升级 PDF 文档解析和分块流水线，提升泛化能力。

## 核心改进领域

### 1. FurnitureDetector（页面装饰元素检测器）
**目标**：替代硬编码正则，使用多特征检测

**新增类：`FurnitureDetector`**
```python
class FurnitureDetector:
    """
    基于多特征的页面装饰元素检测器。
    
    检测策略：
    1. 位置 band 检测：top/bottom 8-10% 区域
    2. 跨页频率统计：高频重复字符串
    3. 特征分类：短文本、数字模式、已知短语
    """
```

**特征列表**：
- `in_edge_band`: 是否在页面边缘区域（y < 10% 或 y > 90%）
- `repeat_ratio`: 该字符串在文档中出现的页面比例
- `is_short`: 文本是否很短（< 5 words）
- `has_page_pattern`: 是否匹配页码模式（纯数字、Page X）
- `has_furniture_phrase`: 是否包含已知装饰短语
- `font_size_delta`: 与正文字体大小的差异

### 2. ProjectionProfileColumnDetector（投影轮廓列检测器）
**目标**：替代简单 x 坐标阈值，使用白空隙检测

**方法**：
1. 计算页面水平投影轮廓
2. 检测空白峡谷（gutter）作为列分隔
3. 将 segments 分配到检测到的列

### 3. Dehyphenation（连字符修复）
**目标**：修复跨行/跨页的连字符断词

**逻辑**：
1. 检测前一段末尾是否有连字符
2. 尝试合并连字符两边的词
3. 验证合并后是否为有效单词

### 4. VisualHeadingLevelInferrer（视觉标题级别推断）
**目标**：不完全依赖 Docling 的 level，基于字体特征自行推断

## 实施顺序

### Phase 1: FurnitureDetector（核心）
1. 创建 FurnitureDetector 类
2. 实现 `scan_document()` - 第一遍扫描建立频率表
3. 实现 `is_furniture()` - 多特征判断
4. 集成到 ReadingOrderCorrector

### Phase 2: 投影轮廓列检测
1. 实现 `detect_columns_by_projection()`
2. 替代 `_phase1_reorder_by_columns` 中的简单阈值

### Phase 3: Dehyphenation
1. 实现 `repair_hyphenation()`
2. 集成到跨页续接检测

### Phase 4: 测试与验证
1. 在 Investments.pdf 上测试
2. 验证边界情况
3. 性能测试

## 代码结构变更

```
logic_segmenter.py
├── FurnitureDetector (NEW)           # 页面装饰检测器
├── ProjectionProfileAnalyzer (NEW)   # 投影轮廓分析
├── DehyphenationHelper (NEW)         # 连字符修复
├── ReadingOrderCorrector (UPDATED)   # 集成新检测器
├── ContinuationDetector (UPDATED)    # 集成连字符修复
└── LogicSegmenter (UPDATED)          # 主流程更新
```

## 配置项新增

```python
class ChunkingConfig:
    # Furniture Detection
    FURNITURE_TOP_BAND = 0.10        # 页面顶部 10%
    FURNITURE_BOTTOM_BAND = 0.10     # 页面底部 10%
    FURNITURE_REPEAT_THRESHOLD = 0.3  # 出现在 30% 以上页面视为重复
    FURNITURE_MAX_WORDS = 5          # 短文本阈值
    
    # Column Detection
    USE_PROJECTION_COLUMN_DETECTION = True
    COLUMN_GUTTER_MIN_WIDTH = 20     # 最小列间隙宽度（px）
    
    # Dehyphenation
    ENABLE_DEHYPHENATION = True
```
