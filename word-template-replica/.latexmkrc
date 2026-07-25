# latexmk 配置文件 —— 强制使用 xelatex 编译
# LaTeX Workshop 和命令行 latexmk 均会读取此文件

$pdf_mode = 5;              # 使用 xelatex 编译（而非 pdflatex）
$xelatex = "xelatex -synctex=1 -interaction=nonstopmode %O %S";
$out_dir = "build";         # 输出文件放入 build 目录
$clean_ext = "aux bbl blg log out fls fdb_latexmk synctex.gz xdv nav snm toc";
