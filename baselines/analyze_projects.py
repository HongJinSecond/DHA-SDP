import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(r'd:\DHA\TrainDatasets.xlsx')

# 统计 project 列
project_counts = df['project'].value_counts()

print(f"数据总行数: {len(df)}")
print(f"不重复项目数: {project_counts.nunique()}")
print(f"\n各项目提交数量统计:")
print("=" * 50)
for project, count in project_counts.items():
    print(f"  {project:<20} {count:>6} 条提交")

# 项目占比
print(f"\n各项目占比:")
print("=" * 50)
for project, count in project_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {project:<20} {pct:>5.1f}%  {bar}")