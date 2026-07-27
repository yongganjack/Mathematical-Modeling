# Q1--Q5 论文图件渲染审阅记录

审阅日期：2026-07-27。所有 PNG 均由 MATLAB 以 300 dpi 导出；同名 PDF 使用 `exportgraphics(..., 'ContentType','vector')` 导出。以下路径均位于 `C:\Users\Jackm\Desktop\matlab_work\2025_guosai\A\figures\paper\`。

| 图件 | 已读 PNG | 最终视觉审阅结论 |
|---|---|---|
| Q1_spatial_strategy | `Q1_spatial_strategy.png` | 中文标题、三维坐标轴和 M1 轨迹清晰；投放/起爆标记在导弹接近区域可辨。 |
| Q1_shielding_diagnosis | `Q1_shielding_diagnosis.png` | 裕度曲线、零边界、遮蔽率和有效时段的浅蓝底纹均无裁切。 |
| Q2_strategy_comparison | `Q2_strategy_comparison.png` | 给定策略以灰色、优化策略以蓝色标记；决策变量改为紧凑文本行，避免右侧裁切。 |
| Q2_convergence_intervals | `Q2_convergence_intervals.png` | PSO/DE 收敛曲线和最终时段条带可读，图例未遮挡主曲线。 |
| Q3_three_bomb_spatial | `Q3_three_bomb_spatial.png` | 三枚烟幕弹的释放与起爆标记可分辨，空间轨迹无裁切。 |
| Q3_interval_contributions | `Q3_interval_contributions.png` | 三条单弹云团时间窗均留出上下边界，未贴图框。 |
| Q3_optimizer_convergence | `Q3_optimizer_convergence.png` | 候选复核由全部候选改为前 8 个，避免高密度柱状图影响读数。 |
| Q4_multi_uav_spatial | `Q4_multi_uav_spatial.png` | FY1--FY3 使用稳定色彩语义，空间部署和 M1 关系清楚。 |
| Q4_timing_contribution | `Q4_timing_contribution.png` | 三段遮蔽与投放—起爆时序均可读，图例位于空白区。 |
| Q4_multistage_convergence | `Q4_multistage_convergence.png` | 初值与分块精修改为同尺度柱状对比，避免原横轴单位不一致。 |
| Q5_multi_target_spatial | `Q5_multi_target_spatial.png` | M1--M3 虚线与五机多弹策略的颜色语义明确，图例未遮挡关键标记。 |
| Q5_target_intervals_duration | `Q5_target_intervals_duration.png` | 分目标甘特图、时长柱及最短保障线清晰；保障注释已移开 M3 柱体。 |
| Q5_three_stage_convergence | `Q5_three_stage_convergence.png` | 三个量纲不同的阶段分成三个并列面板，避免把总时长和最短时长混在同一纵轴。 |

## 导出核验

- 交付目录包含 13 份 PDF、13 份 PNG 和 `manifest.json`。
- 已用 `pdftotext` 读取 `Q5_target_intervals_duration.pdf`；可提取中文标题、坐标标签和数值，表明文本没有整体栅格化。
- MATLAB 对包含三维半透明对象的 PDF 给出“向量化内容可能较慢”的性能提示。该提示未导致导出失败；已逐张读取 PNG，且 PDF 文件大小处于约 12--30 KB 的正常范围，因此保留矢量 PDF 作为论文交付格式。
