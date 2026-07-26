# Q2—Q5 控制台配置选择设计

## 1. 文档目的

规定问题2至问题5启动时选择 `quick.json` 或 `competition.json` 的控制台交互方式，同时保留通过 PyCharm 命令行参数直接指定配置的能力。问题1保持现状。

## 2. 输入信息来源

- 用户确认采用方案1；
- `2025国赛/A/question2/main.py` 至 `question5/main.py`；
- `2025国赛/A/configs/quick.json`；
- `2025国赛/A/configs/competition.json`；
- `2025国赛/A/04_模型求解/代码实现说明.md`。

## 3. 核心内容

### 3.1 控制变量

Q2—Q5 的 `main.py` 分别在模块顶部设置：

```python
USE_CONSOLE_CONFIG_SELECTION = True
```

- `True`：启动后必须通过控制台选择配置，忽略 PyCharm 中的 `--config` 值；
- `False`：不显示菜单，继续使用已有 `--config` 参数，便于 PyCharm、自动测试和批量运行。

### 3.2 控制台菜单

启用交互时显示：

```text
Select configuration:
1. Quick
2. Competition
Enter choice [1/2]:
```

输入 `1` 映射到 `configs/quick.json`，输入 `2` 映射到 `configs/competition.json`。空输入、其他数字或文字均不自动选择，而是提示错误并重新询问。

### 3.3 运行流程

参数解析完成后、调用 `load_config` 前执行选择逻辑。选择完成后把路径写回 `args.config`，其余预算缩放、配置校验、求解、输出和异常处理流程保持不变。

`--validate-config-only` 在交互变量为 `True` 时仍显示菜单；若需要自动配置校验，应临时把变量设为 `False`。

### 3.4 测试范围

只增加最小测试：输入 `1`、输入 `2`、无效输入后重新输入，以及变量为 `False` 时保留命令行配置。Q1 测试与代码不得改变。

## 4. 关键结论

1. Q2—Q5 默认通过控制台选择 quick 或 competition。
2. PyCharm 参数仍由 `USE_CONSOLE_CONFIG_SELECTION=False` 启用。
3. 控制台选择只影响配置路径，不改变模型、算法和输出。
4. Q1 保持现有运行方式。

## 5. 待解决问题

无需要用户继续裁决的问题。

## 6. 与后续步骤的衔接

下一步修改 Q2—Q5 的 `main.py`，提取小型选择函数，运行新增测试和现有相关测试，并更新《代码实现说明》的运行方法。
