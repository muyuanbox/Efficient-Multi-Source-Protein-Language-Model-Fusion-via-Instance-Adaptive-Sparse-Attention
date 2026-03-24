import math

def calc_stats(values):
    n = len(values)
    mean = sum(values) / n

    # 总体方差 / 总体标准差
    var_pop = sum((x - mean) ** 2 for x in values) / n
    std_pop = math.sqrt(var_pop)

    # 样本方差 / 样本标准差
    if n > 1:
        var_sample = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_sample = math.sqrt(var_sample)
    else:
        var_sample = float("nan")
        std_sample = float("nan")

    return mean, var_pop, std_pop, var_sample, std_sample


def main():
    n =6         # 输入行数
    col = 6        # 输入要统计的列号（从1开始）

    values = []
    for _ in range(n):
        line = input().strip()
        parts = line.split()          # 按空白分割（tab/空格都可以）
        values.append(float(parts[col - 1]))

    mean, var_pop, std_pop, var_sample, std_sample = calc_stats(values)

    print(f"第{col}列均值: {mean}")
    print(f"第{col}列总体方差: {var_pop}")
    print(f"第{col}列总体标准差: {std_pop}")
    print(f"第{col}列样本方差: {var_sample}")
    print(f"第{col}列样本标准差: {std_sample}")


if __name__ == "__main__":
    main()