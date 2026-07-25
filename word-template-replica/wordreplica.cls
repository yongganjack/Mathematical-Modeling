% wordreplica.cls — 国赛论文 Word 模版复刻类（增强版）
% 基于 cumcmthesis.cls 的功能体系，保持 Word 模版的视觉精度。
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{wordreplica}[2026/07/25 CUMCM Word template replica]

\LoadClass[UTF8,a4paper,zihao=-4,fontset=none]{ctexart}

% ── 页面布局（逐页比对 Word 导出 PDF 后校准）──────────────────────────────
\RequirePackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm,
  headsep=0.40cm,footskip=2.40cm]{geometry}

% ── 基础宏包 ───────────────────────────────────────────────────────────────
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage{multirow}
\RequirePackage[table]{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage{ragged2e}

% ── 数学 ───────────────────────────────────────────────────────────────────
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}

% ── 表格增强 ───────────────────────────────────────────────────────────────
\RequirePackage{booktabs}
\RequirePackage{bigstrut}

% ── 浮动体控制 ─────────────────────────────────────────────────────────────
\RequirePackage{float}

% ── 列表增强 ───────────────────────────────────────────────────────────────
\RequirePackage{enumitem}
\setlist{%
  topsep=0.3em,
  partopsep=0pt,
  itemsep=0ex plus 0.1ex,
  parsep=0pt,
  leftmargin=1.5em,
  rightmargin=0em,
  labelsep=0.5em,
  labelwidth=2em,
}

% ── 交叉引用与超链接 ───────────────────────────────────────────────────────
\RequirePackage{hyperref}
\hypersetup{%
  pdfstartview=FitH,
  CJKbookmarks=true,
  bookmarksnumbered=true,
  bookmarksopen=true,
  colorlinks,
  allcolors=black,
  breaklinks=true,
}
\RequirePackage{cleveref}
% 中文引用格式
\crefformat{figure}{#2图~#1#3}
\crefrangeformat{figure}{图~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefformat{table}{#2表~#1#3}
\crefrangeformat{table}{表~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefformat{equation}{#2式~(#1#3)}
\crefrangeformat{equation}{式~(#3#1#4)\;\~{}\;(#5#2#6)}

% ── 代码列表 ───────────────────────────────────────────────────────────────
\RequirePackage{listings}
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}
\lstset{%
  frame=tb,
  aboveskip=3mm,
  belowskip=3mm,
  showstringspaces=false,
  columns=flexible,
  framerule=1pt,
  rulecolor=\color{gray!35},
  backgroundcolor=\color{gray!5},
  basicstyle={\small\ttfamily},
  numbers=none,
  keywordstyle=\color{blue},
  commentstyle=\color{dkgreen},
  stringstyle=\color{mauve},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=3,
}

% ── 其它 ───────────────────────────────────────────────────────────────────
\RequirePackage{ulem}       % 下划线
\RequirePackage{setspace}   % 行距

% ── 颜色定义 ───────────────────────────────────────────────────────────────
\definecolor{wordred}{RGB}{255,0,0}
\definecolor{appendixgray}{RGB}{191,191,191}

% ── 字体设置 ───────────────────────────────────────────────────────────────
\setmainfont{Times New Roman}
\setsansfont{Arial}
\newcommand*{\song}{\songti}
\newcommand*{\kai}{\kaishu}

% ── 页码（与原 Word 模版逐页对齐）─────────────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\raisebox{2.00cm}{\zihao{5}\thepage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% ── 段间距 ─────────────────────────────────────────────────────────────────
\setlength{\parindent}{0.8467cm}
\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{1.08}

% ── 图片搜索路径 ───────────────────────────────────────────────────────────
\graphicspath{{figures/}{figure/}{pictures/}{picture/}{pic/}{pics/}{image/}{images/}}

% ── 数学环境：定理类 ───────────────────────────────────────────────────────
\newtheorem{definition}{定义}
\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{assumption}{假设}
\newtheorem{conjecture}{猜想}
\newtheorem{axiom}{公理}
\newtheorem{principle}{定律}
\newtheorem{problem}{问题}
\newtheorem{example}{例}
\newtheorem{proof}{证明}
\newtheorem{solution}{解}

% ── 浮动体设置 ─────────────────────────────────────────────────────────────
\renewcommand*{\textfraction}{0.05}
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}
\renewcommand*{\floatpagefraction}{0.85}
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}

% ═══════════════════════════════════════════════════════════════════════════
%  用户命令（与原 Word 模版一致，不受 ctex 内部机制影响）
% ═══════════════════════════════════════════════════════════════════════════

% ── 论文标题 ───────────────────────────────────────────────────────────────
\newcommand{\wordtitle}[1]{\par{\centering\heiti\zihao{3}#1\par}\vspace{0.35\baselineskip}}

% ── 一级标题（居中、黑体、四号）────────────────────────────────────────────
\newcommand{\mainhead}[1]{\par\vspace{0.20\baselineskip}{\centering\heiti\zihao{4}#1\par}\vspace{0.20\baselineskip}}

% ── 二级标题（左对齐、黑体、小四）──────────────────────────────────────────
\newcommand{\subhead}[1]{\par\vspace{0.10\baselineskip}\noindent\heiti\zihao{-4}#1\par\vspace{0.08\baselineskip}}

% ── 摘要环境 ───────────────────────────────────────────────────────────────
\newenvironment{cumcmabstract}{%
  \begin{center}%
    {\heiti\zihao{4} 摘\quad 要}%
  \end{center}%
  \vspace{0.20\baselineskip}
  \quotation
}{%
  \endquotation
}

% ── 关键词 ────────────────────────────────────────────────────────────────
\newcommand{\cumcmkeywords}[1]{%
  \par\vskip1ex
  {\noindent\heiti 关键词：}~{#1}
}

% ── 红色提示段落 ───────────────────────────────────────────────────────────
\newcommand{\redpara}[1]{\par\begingroup\color{wordred}\justifying\fontsize{12pt}{12pt}\selectfont #1\par\endgroup}

% ── 黑色正文段落 ───────────────────────────────────────────────────────────
\newcommand{\blackpara}[1]{\par\begingroup\justifying\fontsize{12pt}{12pt}\selectfont #1\par\endgroup}

% ── 图注（兼容旧版的手动图注，推荐使用 \caption{}）─────────────────────────
\newcommand{\figcap}[1]{\par{\centering\songti\zihao{5}#1\par}\vspace{0.10\baselineskip}}

% ── 公式占位 ───────────────────────────────────────────────────────────────
\newcommand{\formulaPlaceholder}[1]{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc@{}}
    \songti\zihao{5}这里插入公式 & \songti\zihao{5}(#1)
  \end{tabular*}\par\vspace{1.0\baselineskip}}

% ── 附录项 ────────────────────────────────────────────────────────────────
\newcommand{\appendixbox}[3]{%
  \begin{tabularx}{\textwidth}{|X|}
    \hline
    \rowcolor{appendixgray}\heiti #1 \\ \hline
    \rowcolor{appendixgray}#2 \\ \hline
    \rule{0pt}{#3} \\ \hline
  \end{tabularx}\par\vspace{0.75cm}}

% ── 上标引用 ───────────────────────────────────────────────────────────────
\newcommand{\upcite}[1]{$^{\mbox{\scriptsize \cite{#1}}}$}

\endinput
