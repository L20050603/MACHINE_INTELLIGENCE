# 实验 2：A* 算法求解 8 数码问题

本实验使用 Python 实现 8 数码问题求解器。程序支持任意给定初始状态和目标状态，自动判断是否可解，并输出 A* 搜索得到的移动序列、搜索深度、扩展节点数、生成节点数、最大开放表规模和耗时。

## 实验目标

- 掌握状态空间搜索问题建模方法。
- 使用 A* 算法求解 8 数码问题。
- 设计并比较不同启发函数对搜索效率的影响。
- 加入自主创新：支持更强的线性冲突启发函数、模式数据库 Pattern Database、Weighted A*、随机可解样例生成和启发函数对比实验。

## 文件结构

```text
experiment2/
  eight_puzzle.py    A* 主程序
  web_app.py         朴素前端页面的本地服务
  web/               前端页面
  readme.md          实验说明
  artifacts/         自动生成的 Pattern Database 缓存目录
  __init__.py
```

## 状态表示

棋盘使用 0 表示空格，例如：

```text
2 8 3
1 6 4
7 _ 5
```

对应命令行参数可以写成：

```text
"2 8 3 1 6 4 7 0 5"
```

也可以写成紧凑格式：

```text
283164705
```

## 运行方式

从仓库根目录进入实验 2：

```bash
cd experiment2
python eight_puzzle.py
```

## 图形化交互页面

实验 2 也提供一个朴素的本地页面，用于输入状态、选择启发函数、查看求解统计和逐步展示路径。

启动方式：

```bash
cd experiment2
python web_app.py
```

如果使用指定 Conda 环境：

```bash
cd experiment2
E:\Conda\envs\study\python.exe web_app.py
```

浏览器打开：

```text
http://127.0.0.1:8002
```

页面功能：

- 输入初始状态和目标状态。
- 选择 `misplaced`、`manhattan`、`linear_conflict`、`pattern_db`。
- 显示是否找到解、步数、扩展节点、生成节点、开放表峰值、耗时。
- 用 3x3 棋盘逐步展示移动路径。
- 一键对比四种启发函数。

指定初始状态和目标状态：

```bash
python eight_puzzle.py --start "2 8 3 1 6 4 7 0 5" --goal "1 2 3 8 0 4 7 6 5"
```

只查看统计结果，不打印完整路径：

```bash
python eight_puzzle.py --start "2 8 3 1 6 4 7 0 5" --goal "1 2 3 8 0 4 7 6 5" --no-path
```

生成一个随机可解样例：

```bash
python eight_puzzle.py --random 30 --seed 1
```

## 启发函数

程序提供四种启发函数：

- `misplaced`：错位棋子数，不计算空格。
- `manhattan`：每个棋子到目标位置的曼哈顿距离之和。
- `linear_conflict`：曼哈顿距离 + 线性冲突惩罚。
- `pattern_db`：模式数据库启发函数，默认使用该方法。

指定启发函数：

```bash
python eight_puzzle.py --heuristic manhattan
```

对比四种启发函数：

```bash
python eight_puzzle.py --compare --no-path
```

输出为 CSV 格式：

```text
heuristic,found,depth,expanded,generated,max_frontier,elapsed
misplaced,True,...
manhattan,True,...
linear_conflict,True,...
pattern_db,True,...
```

## 创新点：Pattern Database

模式数据库 Pattern Database 的思想是：不只用一个公式估计距离，而是提前对“部分棋子组合”建立精确距离表。搜索时把当前状态映射到这些抽象模式，再从数据库中读取估计值。

本实验采用两个互不重叠的模式：

```text
Pattern A: 1, 2, 3, 4
Pattern B: 5, 6, 7, 8
```

程序会从目标状态反向搜索，分别预计算两个模式的最短代价。移动属于该模式的棋子时代价为 1，移动其他棋子时代价为 0。由于两个模式互不重叠，两个数据库的估计值可以相加：

```text
h_pdb(n) = pdb_1234(n) + pdb_5678(n)
```

这种做法比曼哈顿距离和线性冲突更有设计性，因为它将“启发函数设计”从手写经验公式提升为“离线预计算 + 在线查询”的组合策略。首次运行 `pattern_db` 时会在 `experiment2/artifacts/` 下生成缓存，后续运行会直接读取缓存。

示例：

```bash
python eight_puzzle.py --heuristic pattern_db --no-path
```

在一次随机样例中，对比结果如下：

```text
heuristic,found,depth,expanded,generated,max_frontier,elapsed
misplaced,True,20,3446,5554,2108,0.018282
manhattan,True,20,419,692,273,0.002705
linear_conflict,True,20,279,451,172,0.003996
pattern_db,True,20,60,108,48,0.096360
```

可以看到 Pattern Database 明显减少了扩展节点数。首次运行耗时会包含建库时间，所以运行时间不一定最短；缓存生成后，再次运行会更快。

## 创新程度对比

| 启发函数 | 说明 | 创新程度 |
| --- | --- | --- |
| `misplaced` | 不在目标位置的数字个数 | 基础 |
| `manhattan` | 每个数字到目标位置的横纵距离之和 | 标准 |
| `linear_conflict` | 在曼哈顿距离基础上增加行/列冲突惩罚 | 较有新意 |
| `pattern_db` | 预先计算部分数字组合的最短距离，搜索时快速查询 | 高分亮点 |

## Weighted A*

标准 A* 使用：

```bash
python eight_puzzle.py --weight 1.0
```

Weighted A* 使用更大的启发函数权重，例如：

```bash
python eight_puzzle.py --weight 1.5 --no-path
```

说明：`weight=1.0` 是标准 A*。当 `weight>1.0` 时，搜索会更偏向启发函数，通常扩展节点更少，但不再保证最优解。这可以作为自主创新部分，用来比较“最优性”和“搜索效率”的权衡。

## 算法说明

A* 对每个状态计算：

```text
f(n) = g(n) + w * h(n)
```

其中：

- `g(n)` 是从初始状态到当前状态的实际步数。
- `h(n)` 是当前状态到目标状态的启发式估计距离。
- `w` 是启发函数权重，标准 A* 中 `w=1.0`。

程序使用优先队列维护开放表，每次取 `f(n)` 最小的状态扩展；同时使用 `best_g` 记录到达每个状态的最短已知代价，避免重复扩展劣质路径。

## 可解性判断

对于 3x3 的 8 数码问题，如果初始状态和目标状态的逆序数奇偶性不同，则不可达。程序会在搜索前进行判断，不可解时直接停止，避免无意义搜索。

示例：

```bash
python eight_puzzle.py --start "1 2 3 4 5 6 8 7 0" --goal "1 2 3 4 5 6 7 8 0"
```

该状态不可解，程序会输出失败原因。

对比命令：

```bash
python eight_puzzle.py --random 40 --seed 7 --compare --no-path
python eight_puzzle.py --random 40 --seed 7 --heuristic linear_conflict --weight 1.0 --no-path
python eight_puzzle.py --random 40 --seed 7 --heuristic linear_conflict --weight 1.5 --no-path
```
