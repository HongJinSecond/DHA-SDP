import pandas as pd
import numpy as np
import os

MODEL_NAME_LIST = ["codebert","plbart", "graphcodebert", "unixcoder",  "codet5","cct5"]
CLUSTER_MODEL_LIST = ["developer_aware","Kmean","project_final"]
N_CLUSTER = 4
BASE_MODEL = "concat"

                    
def result_table(metrics,output_dir):
    for metric in metrics:
        data_table=[]  
        # first run for base
        line=[]
        for model in MODEL_NAME_LIST:
            base_df=pd.read_csv(os.path.join(f"result/Kmean/{str(N_CLUSTER)}/average/{model}", "base_results.csv"),index_col=0)
            base_value=base_df.loc["all"][metric].round(4)
            line.append(base_value)
        data_table.append(line)

        for cluster_model in CLUSTER_MODEL_LIST:
            line=[]
            for model in MODEL_NAME_LIST:
                avg_dir = f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}"
                style_df=pd.read_csv(os.path.join(avg_dir, "outlier_results.csv"),index_col=0)

                style_value=style_df.loc["all"][metric].round(4)
                line.append(style_value)

            data_table.append(line)
        base_df=pd.DataFrame(data_table,index=["base"]+CLUSTER_MODEL_LIST,columns=MODEL_NAME_LIST)
        os.makedirs(output_dir, exist_ok=True)
        base_df.to_csv(os.path.join(output_dir,f"result_table_{metric}.csv"))


def run_avg_base():
    for model in MODEL_NAME_LIST:
        for metric in ["accuracy","precision","recall","recall0","f1","gmean","mcc","auc","R20E","E20R","Popt"]:
            value=0
            for cluster_model in CLUSTER_MODEL_LIST:
                value+=pd.read_csv(os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}", "base_results.csv"),index_col=0).loc["all"][metric]
        
            value = value/3
            for cluster_model in CLUSTER_MODEL_LIST:
            ##### Save mean base on table ####
                df=pd.read_csv(os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}", "base_results.csv"),index_col=0)
                col_index = df.columns.get_loc(metric)
                df.at["all",col_index]=value
                df.to_csv(os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}", "base_results.csv"))  # 保存为 CSV


def RQ1_2_compare():
    for metric in ["gmean","f1","auc"]:
        for cluster_model in CLUSTER_MODEL_LIST:
            data_table=[]
            for model in MODEL_NAME_LIST:
                avg_dir = f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}"
                base_df=pd.read_csv(os.path.join(avg_dir, "base_results.csv"),index_col=0)
                # lora_df=pd.read_csv(os.path.join(avg_dir, "lora_results.csv"),index_col=0)
                style_df=pd.read_csv(os.path.join(avg_dir, "outlier_results.csv"),index_col=0)
                
                model_table=[]
                model_column=[]
                if cluster_model=="developer_aware":
                    target_names=["1","3"]
                elif cluster_model=="project_final": 
                    target_names=["1","2","3","4"]
                else:
                    target_names=["0","1","2","3"]

                for target_name in target_names:
                    base_value=base_df.loc[target_name][metric]
                    style_value=style_df.loc[target_name][metric]
                    model_table.append(base_value)
                    model_table.append(style_value)
                    model_column.append(f"{target_name}_base")
                    model_column.append(f"{target_name}_style")

                data_table.append(model_table)
            cluster_compare_df=pd.DataFrame(data_table,index=MODEL_NAME_LIST,columns=model_column).round(2)
            print(f"{cluster_model}------compare---table-----{metric}")
            print(cluster_compare_df)
            output_path=f"result/RQ1_2"
            os.makedirs(output_path, exist_ok=True)
            cluster_compare_df.to_csv(os.path.join(output_path,f"{cluster_model}_compare_table_{metric}.csv"))

def run_RQ2_compare():
    for metric in ["f1","gmean"]:
        base_lora_data_table=[]
        lora_style_data_table=[]    
        for cluster_model in CLUSTER_MODEL_LIST:
            base_to_lora_line=[]
            lora_to_style_line=[]
            for model in MODEL_NAME_LIST:
                avg_dir = f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}"
                base_df=pd.read_csv(os.path.join(avg_dir, "base_results.csv"),index_col=0)
                lora_df=pd.read_csv(os.path.join(avg_dir, "lora_results.csv"),index_col=0)
                style_df=pd.read_csv(os.path.join(avg_dir, "outlier_results.csv"),index_col=0)
                base_value=base_df.loc["all"][metric].round(2)
                lora_value=lora_df.loc["all"][metric].round(2)
                style_value=style_df.loc["all"][metric].round(2)
                base_to_lora_line.append(f"{base_value}->{lora_value}")
                lora_to_style_line.append(f"{lora_value}->{style_value}")
            base_lora_data_table.append(base_to_lora_line)
            lora_style_data_table.append(lora_to_style_line)
        base_to_lora_df=pd.DataFrame(base_lora_data_table,index=CLUSTER_MODEL_LIST,columns=MODEL_NAME_LIST)
        lora_to_outlier_df=pd.DataFrame(lora_style_data_table,index=CLUSTER_MODEL_LIST,columns=MODEL_NAME_LIST)
        output_path=f"result/RQ2"
        os.makedirs(output_path, exist_ok=True)
        base_to_lora_df.to_csv(os.path.join(output_path,f"base_to_lora_table_{metric}.csv"))
        lora_to_outlier_df.to_csv(os.path.join(output_path,f"lora_to_outlier_table_{metric}.csv"))



def RQ2_2_compare():
    for metric in ["f1"]:            
        for cluster_model in CLUSTER_MODEL_LIST:

            if cluster_model=="developer_aware":
                target_names=["1","3"]
            elif cluster_model=="project_final": 
                target_names=["1","2","3","4"]
            else:
                target_names=["0","1","2","3"]

            base_lora_style_data_table=[]
            for model in MODEL_NAME_LIST:
                base_to_lora_to_style_line=[]
                avg_dir = f"result/{cluster_model}/{str(N_CLUSTER)}/average/{model}"
                base_df=pd.read_csv(os.path.join(avg_dir, "base_results.csv"),index_col=0)
                lora_df=pd.read_csv(os.path.join(avg_dir, "lora_results.csv"),index_col=0)
                style_df=pd.read_csv(os.path.join(avg_dir, "outlier_results.csv"),index_col=0)
            

                for target_name in target_names:
                    base_value=base_df.loc[target_name][metric].round(4)
                    lora_value=lora_df.loc[target_name][metric].round(4)
                    style_value=style_df.loc[target_name][metric].round(4)

                    base_to_lora_to_style_line.append(f"{base_value}->{lora_value}->{style_value}")
                
                base_lora_style_data_table.append(base_to_lora_to_style_line)
            
            
            base_to_lora_to_style_df=pd.DataFrame(base_lora_style_data_table,index=MODEL_NAME_LIST,columns=target_names)
            output_path=f"result/RQ2"
            os.makedirs(output_path, exist_ok=True)
            base_to_lora_to_style_df.to_csv(os.path.join(output_path,f"{cluster_model}_base_to_lora_to_style_table_{metric}.csv"))


    




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_comparison(base_path, lora_path, outlier_path, metrics, output_dir):
    """
    对比可视化函数：生成三组数据的柱状对比图
    
    参数：
    base_path -- 基准数据路径
    lora_path -- LoRA数据路径
    outlier_path -- 异常数据路径
    metrics -- 需要对比的指标列表
    output_dir -- 输出目录路径
    """
    # 设置美观的样式
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.facecolor'] = '0.98'
    
    # 读取数据并设置索引
    base_df = pd.read_csv(base_path, index_col=0)
    lora_df = pd.read_csv(lora_path, index_col=0)
    outlier_df = pd.read_csv(outlier_path, index_col=0)

    # 验证数据一致性
    if not (base_df.index.equals(lora_df.index) and base_df.index.equals(outlier_df.index)):
        raise ValueError("输入数据的索引不一致！")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义更美观的配色
    colors = sns.color_palette("husl", 3)  # 使用seaborn的husl调色板

    # 遍历每个指标
    for metric in metrics:
        # 提取数据
        base = base_df[metric]
        lora = lora_df[metric]
        outlier = outlier_df[metric]

        # 配置绘图参数
        bar_width = 0.25
        index = np.arange(len(base))  # X轴位置
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制三组柱状图（使用新配色）
        bars_base = ax.bar(index - bar_width, base, bar_width, 
                          label='Base', color=colors[0])
        bars_lora = ax.bar(index, lora, bar_width, 
                          label='LoRA', color=colors[1])
        bars_outlier = ax.bar(index + bar_width, outlier, bar_width, 
                            label='Outlier', color=colors[2])

        # 设置坐标轴
        ax.set_xticks(index)
        ax.set_xticklabels(base.index, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Data Index', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'Comparison of {metric} Across Groups', fontsize=13, pad=20)
        ax.legend(frameon=True, shadow=True)

        # 添加数值标签（保留4位小数）
        for bars in [bars_base, bars_lora, bars_outlier]:
            ax.bar_label(bars, padding=3, fontsize=8, 
                        fmt='%.4f')  # 格式化为4位小数

        # 自动调整布局
        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"对比图表已保存至：{output_dir}")



def plot_table():
    for n_cluster in [4]:
        strategy="Isolation"
        n_cluster=n_cluster
        for cluster_model in CLUSTER_MODEL_LIST:
            for pretrained in ["codebert", "graphcodebert", "unixcoder","plbart","codet5","cct5"]:
                base_path=os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/average/{pretrained}", "base_results.csv")
                lora_path=os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/average/{pretrained}", "lora_results.csv")
                outlier_path=os.path.join(f"result/{cluster_model}/{str(N_CLUSTER)}/average/{pretrained}", "outlier_results.csv")
                plot_comparison(base_path,lora_path,outlier_path,
                                metrics=["f1", "gmean","auc","recall"],  # 需要对比的指标
                                output_dir=f"result/{cluster_model}/{str(N_CLUSTER)}/average/{pretrained}",
                                )


def effort_aware_compare_table(metric,output_path,target_file="outlier_results.csv"):
    '''
    Create and save the result table for effort-aware metrics: E20R, R20E and Popt.
    '''
    result_table=[]
    for mode in ["base","developer_aware","Kmean","project_final"]:
        performance=[]
        if mode=="base":
            for model in MODEL_NAME_LIST:
                df_1=pd.read_csv(os.path.join(f"result/Kmean/{str(N_CLUSTER)}/average/{model}", "base_results.csv"),index_col=0)
                df_2=pd.read_csv(os.path.join(f"result/project_final/{str(N_CLUSTER)}/average/{model}", "base_results.csv"),index_col=0)
                df_3=pd.read_csv(os.path.join(f"result/developer_aware/{str(N_CLUSTER)}/average/{model}", "base_results.csv"),index_col=0)

                value_1=df_1.loc["all"][metric]
                value_2=df_2.loc["all"][metric]
                value_3=df_3.loc["all"][metric]

                value=np.mean([value_1,value_2,value_3]).round(4)

                performance.append(value)          
        else:
            for model in MODEL_NAME_LIST:
                df=pd.read_csv(os.path.join(f"result/{mode}/{str(N_CLUSTER)}/average/{model}", target_file),index_col=0)
                value=df.loc["all"][metric].round(4)
                performance.append(value)

        result_table.append(performance)
    ndf=pd.DataFrame(result_table,index=["base","developer_aware","Kmean","project"],columns=MODEL_NAME_LIST)
    ndf.to_csv(output_path)



if __name__=="__main__":
    #Save the table for RQ1
    result_table(["gmean","f1"],"result/RQ1_1")
    RQ1_2_compare()


    #Save the table for RQ2
    run_RQ2_compare()
    RQ2_2_compare()

    #Save the result of RQ3
    effort_aware_compare_table("R20E",f"result/RQ3/compare_R20E.csv")
    effort_aware_compare_table("E20R",f"result/RQ3/compare_E20R.csv")
    effort_aware_compare_table("Popt",f"result/RQ3/compare_Popt.csv")
