import re
import os
import pandas as pd

def extract_metrics_from_log(file_path):
    """
    从日志文件的末尾提取 PIR 纠偏后（或最终评估）的 Test MSE 和 Test MAE。
    新增了对日志末尾极简 'mse:X,mae:Y' 格式的捕获逻辑。
    """
    metrics = {
        'baseline_mse': None, 
        'baseline_mae': None, 
        'corrected_mse': None, 
        'corrected_mae': None  
    }
    
    # 【修复后的最高优先级模式】：匹配日志末尾的极简 'mse:X,mae:Y' 格式。
    # 模式会匹配 'mse:' 后面的数字 (Group 1)，以及 'mae:' 后面的数字 (Group 2)。
    final_metrics_pattern_compact = re.compile(
        r"mse:(\d+\.\d+).*?mae:(\d+\.\d+)", 
        re.IGNORECASE | re.DOTALL
    )

    # 兜底模式（如果最高优先级失败，则用于捕获最新的 Test Loss MSE，但不含 MAE）
    epoch_pattern = re.compile(r"Epoch: (\d+).*Vali Loss: (\d+\.\d+).*Test Loss: (\d+\.\d+)")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        # --- 优先级 1: 捕获日志末尾的简洁 MSE/MAE (最高优先级) ---
        # 从文件末尾向上查找（最后 5000 个字符）
        final_matches = final_metrics_pattern_compact.findall(log_content[-5000:])
        if final_matches:
            # 找到最后一个匹配项作为 PIR/Refine 后的最终结果 (即 mse:0.219,mae:0.316)
            # 注意: final_matches[-1] 返回的是 (mse_str, mae_str)
            mse_val, mae_val = final_matches[-1] 
            metrics['corrected_mse'] = float(mse_val)
            metrics['corrected_mae'] = float(mae_val)
            return metrics # 找到了明确的 MSE/MAE，直接返回

        # --- 优先级 2: 如果没有找到简洁格式，则提取最后一行的 Test Loss (仅 MSE) ---
        all_epochs_matches = epoch_pattern.findall(log_content)
        if all_epochs_matches:
            # 提取最后一个 Test Loss (MSE) 作为最新的结果
            latest_test_loss = float(all_epochs_matches[-1][2]) # [2] 是 Test Loss
            metrics['corrected_mse'] = latest_test_loss
            # MAE 仍然缺失，保持 None
            
        return metrics
    except Exception as e:
        print(f"读取或解析文件 '{file_path}' 时出错: {e}")
        return metrics 

# --- parse_filename 函数保持不变，以适应 PIR_Electricity_timesnet_pl720_dm256_sl96.log ---
def parse_filename(filename):
    """
    从文件名中解析出 数据集, 模型, 和 预测步长 (Horizon)。
    """
    try:
        parts = filename.replace('.log', '').split('_')
        dataset = parts[1] 
        model = f"{parts[0]}_{parts[2]}" 
        horizon = None
        for part in parts:
            if part.startswith('pl'):
                horizon = int(part[2:])
                break
        return dataset, model, horizon
    except Exception as e:
        print(f"  - 警告: 无法解析文件名 '{filename}'。跳过。错误: {e}")
        return None, None, None

# --- process_log_directory 函数保持不变 ---
def process_log_directory(root_directory):
    if not os.path.isdir(root_directory):
        print(f"错误：目录 '{root_directory}' 不存在。")
        return

    all_results = []
    for filename in os.listdir(root_directory):
        if filename.endswith(".log"):
            print(f"正在处理文件: {filename}...")
            dataset, model, horizon = parse_filename(filename)
            if not dataset: 
                continue

            full_path = os.path.join(root_directory, filename)
            metrics = extract_metrics_from_log(full_path)
            
            if metrics:
                result_row = {
                    'dataset': dataset, 'model': model, 'horizon': horizon,
                    'baseline_mse': metrics['baseline_mse'], 'baseline_mae': metrics['baseline_mae'],
                    'corrected_mse': metrics['corrected_mse'], 'corrected_mae': metrics['corrected_mae'],
                    'filename': filename
                }
                all_results.append(result_row)

    if all_results:
        df = pd.DataFrame(all_results)
        cols_order = ['dataset', 'model', 'horizon', 'corrected_mse', 'corrected_mae', 'baseline_mse', 'baseline_mae', 'filename']
        existing_cols = [col for col in cols_order if col in df.columns]
        df = df[existing_cols]
        
        print("\n" + "="*30 + " 结 果 汇 总 " + "="*30)
        print(df) 
        
        output_csv_path = os.path.join(root_directory, 'results_summary_detailed.csv')
        try:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n结果已成功保存到: {output_csv_path}")
        except Exception as e:
            print(f"\n保存CSV文件时出错: {e}")
            
    else:
        print("\n在指定目录中没有找到任何有效的日志文件或性能指标。")

# --- 使用示例 ---

# !! 重要：请将这里的路径替换成你自己的文件夹路径 !!
log_directory_path = 'logs/multi_PIR_20251120_134638'
process_log_directory(log_directory_path)
