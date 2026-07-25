# Word 国赛论文模板 LaTeX 复刻版

## 简介

本模板是对 **1.国赛论文模版.docx**（全国大学生数学建模竞赛论文 Word 模板）的严格 LaTeX 复刻版本。从文字排版、图片位置、分页断点到页边距，均已与原 Word 模板逐页比对校准，确保编译输出的 PDF 与原 Word 文档的视觉效果高度一致。

模板底层参考了经典的 `cumcmthesis.cls` 文档类体系，在此基础上重新实现了页面布局与排版命令，同时扩展了定理环境、代码列表、交叉引用等功能。

## 文件夹结构

```
word-template-replica/
├── main.tex                  ← 主文档，论文内容在此编写
├── settings.tex              ← 字体配置文件
├── wordreplica.cls           ← 文档类文件（定义页面布局与排版命令）
├── .latexmkrc                ← latexmk 编译配置文件（自动调用 xelatex）
├── .vscode/
│   └── settings.json         ← VS Code LaTeX Workshop 插件预设配置
├── figures/                  ← 图片资源文件夹
│   ├── figure1.png           ← 总体思路图
│   ├── figure2.png           ← 问题一模型图
│   ├── figure3.png           ← 问题二模型图
│   └── figure4.png           ← 问题三模型图
├── build/                    ← 编译输出文件夹
│   ├── main.pdf              ← 编译生成的 PDF 论文
│   └── main.bbl              ← 参考文献编译中间文件
├── 数学建模国赛latex模板/     ← 参考：原版 cumcmthesis 模板（含示例 PDF）
│   ├── cumcmthesis.cls
│   ├── example.tex
│   ├── example.pdf
│   └── figures/
└── README.md                 ← 本说明文档
```

## 环境要求

- **TeX 发行版**：TeX Live 2024 或更新版本（需包含 `xelatex` 编译器）
- **latexmk**（推荐）：通常随 TeX Live 一同安装，用于自动化编译流程
- **中文字体（Windows）**：宋体（`simsun.ttc`）、黑体（`simhei.ttf`）、仿宋（`simfang.ttf`）——默认已安装在 `C:/Windows/Fonts/`
- **中文字体（macOS/Linux）**：`settings.tex` 已提供 Fandol 字体的备用配置，也可替换为自己系统上已安装的字体
- **编辑器**：VS Code（推荐装 LaTeX Workshop 插件）、TeXstudio、WinEdt 等均可

## 快速开始

### 1. 编译（推荐使用 latexmk）

项目根目录已配置 `.latexmkrc`，可一键编译：

```bash
latexmk
```

`latexmk` 会自动调用 `xelatex` 并处理交叉引用和参考文献的多次编译，输出文件统一放在 `build/` 目录。

### 2. 手动编译

也可以直接调用 `xelatex`：

```bash
xelatex -output-directory=build main.tex
```

> **注意**：必须使用 `xelatex` 编译，`pdflatex` 不支持中文字体。

### 3. 在 VS Code 中编译

项目已预置 `.vscode/settings.json`，安装了 LaTeX Workshop 插件后即可使用：

- 打开 `main.tex` → 左侧 TeX 面板 → **Build LaTeX project**
- 默认使用 `xelatex (latexmkrc)` 配方，也可在插件面板中切换到其他配方
- 编译完成后，PDF 预览自动出现在编辑器右侧

### 4. 查看输出

编译成功后，PDF 文件生成在 `build/main.pdf`。

## 如何编写论文

### 文档结构概览

`main.tex` 已经按论文顺序排好了所有章节，你可以对照以下结构直接填空：

| 页码 | 内容 |
|------|------|
| 第 1 页 | 题目、摘要、关键词 |
| 第 2 页 | 问题重述（1.1 问题背景、1.2 题目信息、1.3 待求解问题） |
| 第 3–4 页 | 问题分析、总体思路图、模型假设、符号说明、问题一模型 |
| 第 5 页 | 问题二、问题三、公式占位区 |
| 第 6 页 | 模型评价与改进、参考文献 |
| 第 7 页 | 附录（支撑材料清单、核心代码、程序源代码） |

### 常用命令说明

模板在 `wordreplica.cls` 中定义了一系列排版命令，在 `main.tex` 中直接使用即可：

| 命令 | 用途 | 示例 |
|------|------|------|
| `\wordtitle{标题}` | 论文大标题（黑体三号居中） | `\wordtitle{基于XXX模型的XXXX问题研究}` |
| `\mainhead{标题}` | 一级标题（黑体四号居中，如"一、问题重述"） | `\mainhead{一、问题重述}` |
| `\subhead{标题}` | 二级标题（黑体小四左对齐，如"1.1 问题背景"） | `\subhead{1.\ 1 问题背景}` |
| `\blackpara{文字}` | 摘要正文段落（黑色，12pt） | `\blackpara{开头段概括论文内容...}` |
| `\keywordline{关键词}` | 关键词行（黑体标签 + 正文关键词） | `\keywordline{关键词1\quad 关键词2\quad 关键词3}` |
| `\redpara{文字}` | 红色提示段落（用于模板说明，写正式论文时替换为正文） | `\redpara{这部分的内容是...}` |
| `\figcap{图注}` | 图片标题（宋体五号居中；推荐直接使用 `\caption{}`） | `\figcap{图1\quad 总体思路图}` |
| `\formulaPlaceholder{n}` | 公式占位符（编号 n，正式论文替换为实际公式） | `\formulaPlaceholder{1}` |
| `\appendixbox{标题}{描述}{高度}` | 附录条目（灰底表格框） | `\appendixbox{附录1}{支撑材料文件列表}{2.30cm}` |
| `\upcite{key}` | 上标引用（显示在文字右上角） | `\upcite{liuhaiyang2013latex}` |

### 智能交叉引用（`\cref{}`）

模板已配置 `cleveref` 宏包的中文引用格式，引用图表和公式时无需手动写"图""表""式"前缀：

```latex
如\cref{fig:overview} 所示，……           % → 如"图 1"所示
\cref{tab:comparison} 展示了不同参数……   % → "表 1"展示了不同参数
式\cref{eq:objective} 为目标函数……       % → 式"(1)"为目标函数
```

支持的标签类型：`figure`、`table`、`equation`，自动添加中文前缀。

### 数学公式

推荐使用 `equation` 环境编写带编号的公式（支持 `\label` + `\cref` 交叉引用）：

```latex
\begin{equation}
  \min\ f(\bm{X}) = \sum_{i=1}^{n} c_i x_i
  \label{eq:objective}
\end{equation}
```

模板已加载 `amsmath`、`amsfonts`、`amssymb`、`bm`（加粗数学符号）等宏包。

### 定理环境

`wordreplica.cls` 预定义了以下定理类环境，可直接使用：

| 环境 | 用途 |
|------|------|
| `theorem` | 定理 |
| `lemma` | 引理 |
| `corollary` | 推论 |
| `definition` | 定义 |
| `assumption` | 假设 |
| `proof` | 证明 |
| `example` | 例 |
| `problem` | 问题 |
| `solution` | 解 |

示例：

```latex
\begin{theorem}[最优解存在性]
  若目标函数 $f(\bm{X})$ 为连续函数且可行域为非空紧集，
  则优化问题至少存在一个全局最优解。
\end{theorem}
```

### 代码列表

使用 `lstlisting` 环境展示程序代码（已预设 Python 语法高亮）：

```latex
\begin{lstlisting}[language=Python]
import numpy as np
# 你的代码...
\end{lstlisting}
```

模板预设了蓝色关键字、绿色注释、紫色字符串的配色方案，浅灰背景带框线。

### 三线表

使用 `booktabs` 宏包的三线表格式：

```latex
\begin{table}[H]
  \centering
  \caption{表格标题}
  \label{tab:xxx}
  \begin{tabular}{ccc}
    \toprule[1.5pt]
    列1 & 列2 & 列3 \\
    \midrule[1pt]
    数据 & 数据 & 数据 \\
    \bottomrule[1.5pt]
  \end{tabular}
\end{table}
```

### 写作流程

1. **替换标题**：将 `\wordtitle{}` 中的占位标题改为你的论文题目
2. **撰写摘要**：将 `\blackpara{}` 中的红色提示替换为你的摘要正文
3. **修改关键词**：在 `\cumcmkeywords{}` 中填入你的关键词
4. **逐章编写**：
   - 问题重述 → 用自己的话复述题目
   - 问题分析 → 分析每个问题的求解思路
   - 模型假设 → 列出建模假设条件
   - 符号说明 → 在表格中填入主要符号
   - 模型建立与求解 → 写核心内容，插入公式和图表
   - 模型评价与改进 → 总结优缺点
   - 参考文献 → 按规范格式列出
   - 附录 → 列出支撑材料与代码
5. **替换图片**：将 `figures/` 中的占位图片替换为你自己的截图或图表
6. **红色提示文字**：模板中所有 `\redpara{}` 包裹的内容均为写作指导提示，正式提交前应删除或替换为正文

## 自定义字体

如果你的系统中文字体路径不同，请编辑 `settings.tex`，修改字体路径：

```latex
% Windows 用户（默认）
\setCJKmainfont[Path=C:/Windows/Fonts/]{simsun.ttc}

% macOS 用户示例
% \setCJKmainfont[Path=/System/Library/Fonts/]{Songti.ttc}

% Linux 用户可使用 Fandol 字体（settings.tex 已自带备用配置）
```

## 注意事项

1. **编译器**：务必使用 `xelatex`，其他编译器无法正确处理中文字体
2. **红字提示**：`\redpara{}` 中的红色文字是写作指导，正式提交论文前必须删除或替换
3. **公式替换**：`\formulaPlaceholder{}` 是公式占位符，正式论文中应替换为实际的数学公式（推荐使用 `equation` 环境）
4. **图片替换**：`figures/` 中的四张图片是示意图，请替换为你自己的模型图
5. **分页控制**：模板的分页位置已与原 Word 对齐；若增删内容导致分页变化，可调整 `\clearpage` 和 `\vspace{}` 参数
6. **查重提醒**：问题重述部分**不可照抄原题**，必须用自己的语言重新描述
7. **latexmk 清理**：如需清理临时文件，运行 `latexmk -c`；彻底清理（含 PDF）运行 `latexmk -C`

## 参考资源

- `数学建模国赛latex模板/` — 原版 `cumcmthesis.cls` 模板，含完整示例 PDF，可作为补充参考
- [全国大学生数学建模竞赛组委会](https://www.mcm.edu.cn) — 官方竞赛通知与论文格式规范
- LaTeX 入门推荐：[刘海洋. LaTeX 入门. 电子工业出版社, 2013]
