import os
import json
import pandas as pd
import glob
import argparse
import numpy as np
from datetime import datetime

CATEGORY_NAMES = {
    1: "多跳问题(Multi-hop)",
    2: "时间问题(Temporal)",
    3: "开放域问题(Open-domain)",
    4: "单跳问题(Single-hop)",
    5: "对抗性问题(Adversarial)",
}

def load_json_data(input_dir):
    """Load json data."""
    if not os.path.exists(input_dir):
        print(f" 错误: 输入目录 '{input_dir}' 不存在。")
        return [], {}

    patterns = ["*.json"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    files = list(set(files))
    
    all_details = []
    sample_meta_map = {}
    
    print(f" 正在扫描目录: {os.path.abspath(input_dir)}")
    print(f" 找到 {len(files)} 个 JSON 文件")

    summary_files = [f for f in files if "final_summary" in os.path.basename(f)]
    if summary_files:
        print(f" 发现汇总文件: {os.path.basename(summary_files[0])}")
        try:
            with open(summary_files[0], 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                if "sample_summaries" in summary_data:
                    for sid, s_info in summary_data["sample_summaries"].items():
                        count = s_info.get("successful_count", s_info.get("test_count", 0))
                        sample_meta_map[sid] = count
        except Exception as e:
            print(f" 读取汇总文件失败: {e}")

    valid_files_count = 0
    for file_path in files:
        fname = os.path.basename(file_path)
        if fname.startswith("final_summary") or fname.startswith("report"):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                results = []
                sample_id = "unknown"
                
                if isinstance(data, dict):
                    sample_info = data.get("sample_info", {})
                    sample_id = sample_info.get("sample_id", fname.replace(".json", ""))
                    
                    if sample_id not in sample_meta_map and sample_id != "unknown":
                        sample_meta_map[sample_id] = sample_info.get('successful_count', 0)
                    
                    for key in ["results", "detailed_results", "qa_results", "data"]:
                        if key in data and isinstance(data[key], list):
                            results = data[key]
                            break
                elif isinstance(data, list):
                    results = data
                    sample_id = fname.replace(".json", "")

                if not results:
                    continue

                valid_files_count += 1
                for item in results:
                    item['_source_file'] = fname
                    if 'sample_id' not in item:
                        item['sample_id'] = sample_id
                    
                    try:
                        item['category'] = int(item.get('category', 0))
                    except:
                        item['category'] = 0

                    if 'evaluation_scores' in item and isinstance(item['evaluation_scores'], dict):
                        scores = item.pop('evaluation_scores')
                        for k, v in scores.items():
                            item[f'score_{k}'] = v
                    
                    item['tokens_input'] = item.get('total_input_tokens', 0)
                    item['tokens_output'] = item.get('completion_tokens', 0)
                    item['tokens_graph'] = item.get('graph_tokens', 0)
                    item['tokens_episodic'] = item.get('episodic_tokens', 0)
                    item['tokens_hierarchical'] = item.get('hierarchical_tokens', 0)
                    
                    if item['tokens_input'] == 0:
                        item['tokens_input'] = item.get('system_prompt_tokens', 0) + item.get('question_tokens', 0) + \
                                               item['tokens_graph'] + item['tokens_episodic'] + item['tokens_hierarchical']

                    if 'retrieval_details' in item and isinstance(item['retrieval_details'], dict):
                        rd = item.pop('retrieval_details')
                        item['retrieval_method'] = rd.get('method', 'unknown')

                    all_details.append(item)

        except Exception as e:
            print(f" 读取文件 {file_path} 失败: {e}")
            
    print(f" 已处理 {valid_files_count} 个详细结果文件")
    print(f" 已加载 {len(all_details)} 条详细记录")
    return all_details, sample_meta_map


def create_excel_report(details_data, sample_meta_map, output_file):
    if not details_data:
        print(" 无有效数据，无法生成报告")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f" 已创建输出目录: {output_dir}")
        except Exception as e:
            print(f" 无法创建目录 {output_dir}: {e}")
            return

    df = pd.DataFrame(details_data)
    
    if 'final_answer' not in df.columns and 'generated_answer' in df.columns:
        df['final_answer'] = df['generated_answer']

    score_cols = ['score_llm_accuracy', 'score_token_f1', 'score_semantic_similarity', 'score_exact_match']
    time_cols = ['total_retrieval_time', 'generation_time', 'end_to_end_latency',
                 'hierarchical_time', 'graph_time', 'episodic_time']
    token_cols = ['tokens_input', 'tokens_output', 'tokens_graph', 'tokens_episodic', 'tokens_hierarchical']
    
    for col in score_cols + time_cols + token_cols:
        if col not in df.columns: df[col] = 0

    if 'category' in df.columns:
        df['category'] = df['category'].map(lambda x: CATEGORY_NAMES.get(x, f"未知({x})"))

    if 'sample_id' in df.columns:
        df.sort_values(by=['sample_id', 'category'], inplace=True)

    adv_name = CATEGORY_NAMES.get(5, '对抗性问题(Adversarial)')

    
    
    df_no_adv = df[df['category'] != adv_name]
    
    if sample_meta_map:
        total_questions = sum(sample_meta_map.values())
    else:
        total_questions = len(df)

    adv_count_in_df = len(df[df['category'] == adv_name])
    non_adv_total = total_questions - adv_count_in_df

    dashboard_data = [
        {'Metric (指标)': 'Total Questions (总问题数 - 基于元数据)', 'Value': total_questions},
        {'Metric (指标)': 'Unique Samples (样本数)', 'Value': df['sample_id'].nunique() if 'sample_id' in df.columns else 1},
        {'Metric (指标)': '--------------------------------', 'Value': ''},
        {'Metric (指标)': 'Avg LLM Accuracy [All] (平均LLM准确率-含对抗)', 
        'Value': df['score_llm_accuracy'].fillna(0).sum() / total_questions if total_questions > 0 else 0},

        {'Metric (指标)': 'Avg LLM Accuracy [No Adv] (平均LLM准确率-除对抗)', 
        'Value': df_no_adv['score_llm_accuracy'].fillna(0).sum() / non_adv_total if non_adv_total > 0 else 0}, 
        {'Metric (指标)': '--------------------------------', 'Value': ''},
        {'Metric (指标)': 'Avg F1-Score (平均F1分数)', 'Value': df['score_token_f1'].mean()},
        {'Metric (指标)': 'Avg Semantic Sim (语义相似度)', 'Value': df['score_semantic_similarity'].mean()},
        {'Metric (指标)': 'Avg Exact Match (完全匹配)', 'Value': df['score_exact_match'].mean()},
        {'Metric (指标)': '--------------------------------', 'Value': ''},
        {'Metric (指标)': 'Avg Input Tokens (平均输入Token)', 'Value': df['tokens_input'].mean()},
        {'Metric (指标)': 'Avg Retrieval Time (平均检索时间 s)', 'Value': df['total_retrieval_time'].mean()},
    ]
    df_dashboard = pd.DataFrame(dashboard_data)

    
    # Dataset-specific handling used by the reproduction workflow.
    
    cat_short_names = {
        CATEGORY_NAMES.get(1, ''): 'Multi-hop',
        CATEGORY_NAMES.get(2, ''): 'Temporal',
        CATEGORY_NAMES.get(3, ''): 'Open-domain',
        CATEGORY_NAMES.get(4, ''): 'Single-hop',
        CATEGORY_NAMES.get(5, ''): 'Adversarial',
    }
    df_sample_acc = pd.DataFrame()
    if 'sample_id' in df.columns:
        def calc_acc(x):
            sid = x.name
            real_count = sample_meta_map.get(sid, len(x))
            adv_rows_in_sample = len(x[x['category'] == adv_name])
            non_adv_count = real_count - adv_rows_in_sample
            non_adv_data = x[x['category'] != adv_name]
            result = {
                'Count (Meta)': real_count,
                'LLM Acc (All)': x['score_llm_accuracy'].fillna(0).sum() / real_count if real_count > 0 else 0,
                'LLM Acc (No Adv)': non_adv_data['score_llm_accuracy'].fillna(0).sum() / non_adv_count if non_adv_count > 0 else 0,
            }
            for cat_id in sorted(CATEGORY_NAMES.keys()):
                cat_full = CATEGORY_NAMES[cat_id]
                cat_short = cat_short_names.get(cat_full, cat_full)
                cat_data = x[x['category'] == cat_full]
                cat_n = len(cat_data)
                result[f'{cat_short} (#)'] = cat_n
                result[f'{cat_short} (Acc)'] = cat_data['score_llm_accuracy'].fillna(0).sum() / cat_n if cat_n > 0 else 0
            return pd.Series(result)
        df_sample_acc = df.groupby('sample_id').apply(calc_acc).reset_index()

    
    # Sheet 3: Category_Accuracy
    
    df_cat_acc = pd.DataFrame()
    if 'category' in df.columns:
        def calc_cat_acc(x):
            n = len(x)
            return pd.Series({
                'Count': n,
                'LLM Accuracy': x['score_llm_accuracy'].fillna(0).sum() / n if n > 0 else 0,
                'Token F1': x['score_token_f1'].fillna(0).sum() / n if n > 0 else 0,
                'Semantic Similarity': x['score_semantic_similarity'].fillna(0).sum() / n if n > 0 else 0,
                'Exact Match': x['score_exact_match'].fillna(0).sum() / n if n > 0 else 0,
            })
        df_cat_acc = df.groupby('category').apply(calc_cat_acc).reset_index()

    
    # Sheet 4: Time_Stats
    
    target_times = [c for c in time_cols if c in df.columns]
    agg_time = {c: 'mean' for c in target_times}
    agg_time['question'] = 'count'
    
    df_time_sample = pd.DataFrame()
    df_time_cat = pd.DataFrame()
    if 'sample_id' in df.columns and 'question' in df.columns:
        df_time_sample = df.groupby('sample_id').agg(agg_time).reset_index().rename(columns={'question': 'Computed Count'})
    if 'category' in df.columns and 'question' in df.columns:
        df_time_cat = df.groupby('category').agg(agg_time).reset_index().rename(columns={'question': 'Computed Count'})

    
    # Sheet 5: Token_Stats
    
    target_tokens = [c for c in token_cols if c in df.columns]
    agg_token = {c: 'mean' for c in target_tokens}
    agg_token['question'] = 'count'
    
    df_token_sample = pd.DataFrame()
    df_token_cat = pd.DataFrame()
    if 'sample_id' in df.columns and 'question' in df.columns:
        df_token_sample = df.groupby('sample_id').agg(agg_token).reset_index().rename(columns={'question': 'Computed Count'})
    if 'category' in df.columns and 'question' in df.columns:
        df_token_cat = df.groupby('category').agg(agg_token).reset_index().rename(columns={'question': 'Computed Count'})

    
    # Sheet 6: Detailed_Logs
    
    base_cols = ['sample_id', 'category', 'question', 'final_answer', 'expected_answer']
    final_cols = [c for c in (base_cols + score_cols + time_cols + token_cols) if c in df.columns]
    df_details = df[final_cols]

    print(f" 正在生成 Excel: {output_file}")
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    workbook = writer.book

    center_base = {'align': 'center', 'valign': 'vcenter'}
    header_fmt = workbook.add_format({**center_base, 'bold': True, 'fg_color': '#D7E4BC', 'border': 1, 'text_wrap': True})
    title_fmt = workbook.add_format({'bold': True, 'font_size': 12, 'font_color': '#333333', 'bg_color': '#F2F2F2', 'valign': 'vcenter'})
    num_fmt = workbook.add_format({**center_base, 'num_format': '0.000'})
    int_fmt = workbook.add_format({**center_base, 'num_format': '0'})
    percent_fmt = workbook.add_format({**center_base, 'num_format': '0.00%'})
    text_center_fmt = workbook.add_format({**center_base, 'text_wrap': True})
    text_left_fmt = workbook.add_format({'align': 'left', 'valign': 'vcenter', 'text_wrap': True})
    green_bg = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'align': 'center', 'valign': 'vcenter'})
    red_bg = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'align': 'center', 'valign': 'vcenter'})

    # --- Sheet 1: Dashboard ---
    df_dashboard.to_excel(writer, sheet_name='Dashboard', index=False)
    ws = writer.sheets['Dashboard']
    ws.set_column('A:A', 45, text_left_fmt)
    ws.set_column('B:B', 20, num_fmt)
    for col_num, value in enumerate(df_dashboard.columns.values):
        ws.write(0, col_num, value, header_fmt)
    ws.write(4, 1, df_dashboard.iloc[3]['Value'], percent_fmt)  # Avg LLM Accuracy [All]
    ws.write(5, 1, df_dashboard.iloc[4]['Value'], percent_fmt)  # Avg LLM Accuracy [No Adv]

    # --- Sheet 2: Sample_LLM_Stats ---
    if not df_sample_acc.empty:
        df_sample_acc.to_excel(writer, sheet_name='Sample_LLM_Stats', index=False)
        ws = writer.sheets['Sample_LLM_Stats']
        ws.set_column('A:A', 15, text_center_fmt)
        ws.set_column('B:B', 15, int_fmt)
        ws.set_column('C:D', 20, percent_fmt)
        col_idx = 4
        for _ in sorted(CATEGORY_NAMES.keys()):
            ws.set_column(col_idx, col_idx, 12, int_fmt)
            ws.set_column(col_idx + 1, col_idx + 1, 16, percent_fmt)
            col_idx += 2
        for col_num, value in enumerate(df_sample_acc.columns.values):
            ws.write(0, col_num, value, header_fmt)
        ws.conditional_format('C2:D100', {'type': '3_color_scale'})
        acc_col_idx = 5
        for _ in sorted(CATEGORY_NAMES.keys()):
            col_letter = chr(ord('A') + acc_col_idx) if acc_col_idx < 26 else None
            if col_letter:
                ws.conditional_format(f'{col_letter}2:{col_letter}100', {'type': '3_color_scale'})
            acc_col_idx += 2

    # --- Sheet 3: Category_Accuracy ---
    if not df_cat_acc.empty:
        df_cat_acc.to_excel(writer, sheet_name='Category_Accuracy', index=False)
        ws = writer.sheets['Category_Accuracy']
        ws.set_column('A:A', 28, text_left_fmt)
        ws.set_column('B:B', 12, int_fmt)
        ws.set_column('C:F', 20, percent_fmt)
        for col_num, value in enumerate(df_cat_acc.columns.values):
            ws.write(0, col_num, value, header_fmt)
        ws.conditional_format('C2:F100', {'type': '3_color_scale'})

    # --- Sheet 4: Time_Stats ---
    ws = workbook.add_worksheet('Time_Stats')
    writer.sheets['Time_Stats'] = ws
    row = 0
    if not df_time_sample.empty:
        ws.write(row, 0, "按样本平均耗时 (Average Time per Sample) [Seconds]", title_fmt)
        row += 1
        for col_num, val in enumerate(df_time_sample.columns): ws.write(row, col_num, val, header_fmt)
        for r_idx, r_val in enumerate(df_time_sample.values):
            ws.write(row+1+r_idx, 0, r_val[0], text_center_fmt)
            for c_idx, cell_val in enumerate(r_val[1:], 1):
                fmt = int_fmt if "Count" in df_time_sample.columns[c_idx] else num_fmt
                ws.write(row+1+r_idx, c_idx, cell_val, fmt)
        row += len(df_time_sample) + 3

    if not df_time_cat.empty:
        ws.write(row, 0, "按类别平均耗时 (Average Time per Category) [Seconds]", title_fmt)
        row += 1
        for col_num, val in enumerate(df_time_cat.columns): ws.write(row, col_num, val, header_fmt)
        for r_idx, r_val in enumerate(df_time_cat.values):
            ws.write(row+1+r_idx, 0, r_val[0], text_left_fmt)
            for c_idx, cell_val in enumerate(r_val[1:], 1):
                fmt = int_fmt if "Count" in df_time_cat.columns[c_idx] else num_fmt
                ws.write(row+1+r_idx, c_idx, cell_val, fmt)
    ws.set_column(0, 0, 28)
    ws.set_column(1, 20, 18)

    # --- Sheet 5: Token_Stats ---
    ws = workbook.add_worksheet('Token_Stats')
    writer.sheets['Token_Stats'] = ws
    row = 0
    if not df_token_sample.empty:
        ws.write(row, 0, "按样本平均Token (Average Tokens per Sample)", title_fmt)
        row += 1
        for col_num, val in enumerate(df_token_sample.columns): ws.write(row, col_num, val, header_fmt)
        for r_idx, r_val in enumerate(df_token_sample.values):
            ws.write(row+1+r_idx, 0, r_val[0], text_center_fmt)
            for c_idx, cell_val in enumerate(r_val[1:], 1): ws.write(row+1+r_idx, c_idx, cell_val, int_fmt)
        row += len(df_token_sample) + 3
    
    if not df_token_cat.empty:
        ws.write(row, 0, "按类别平均Token (Average Tokens per Category)", title_fmt)
        row += 1
        for col_num, val in enumerate(df_token_cat.columns): ws.write(row, col_num, val, header_fmt)
        for r_idx, r_val in enumerate(df_token_cat.values):
            ws.write(row+1+r_idx, 0, r_val[0], text_left_fmt)
            for c_idx, cell_val in enumerate(r_val[1:], 1): ws.write(row+1+r_idx, c_idx, cell_val, int_fmt)
    ws.set_column(0, 0, 28)
    ws.set_column(1, 20, 18)

    # --- Sheet 6: Detailed_Logs ---
    df_details.to_excel(writer, sheet_name='Detailed_Logs', index=False)
    ws = writer.sheets['Detailed_Logs']
    
    for col_num, value in enumerate(df_details.columns.values):
        ws.write(0, col_num, value, header_fmt)

    ws.set_column('A:A', 15, text_center_fmt)
    ws.set_column('B:B', 28, text_left_fmt)
    ws.set_column('C:E', 50, text_left_fmt)
    
    start_num_col = 5
    ws.set_column(start_num_col, len(final_cols)-1, 12, num_fmt)
    
    try:
        acc_idx = final_cols.index('score_llm_accuracy')
        acc_col_letter = chr(65 + acc_idx)
        ws.conditional_format(f'{acc_col_letter}2:{acc_col_letter}{len(df_details)+1}', {
            'type': 'cell', 'criteria': '==', 'value': 1, 'format': green_bg
        })
        ws.conditional_format(f'{acc_col_letter}2:{acc_col_letter}{len(df_details)+1}', {
            'type': 'cell', 'criteria': '==', 'value': 0, 'format': red_bg
        })
    except: pass

    try:
        f1_idx = final_cols.index('score_token_f1')
        f1_col = chr(65 + f1_idx)
        ws.conditional_format(f'{f1_col}2:{f1_col}{len(df_details)+1}', {'type': '3_color_scale'})
    except: pass

    ws.freeze_panes(1, 0)
    
    writer.close()
    print(f" Re-Eval Excel 报告生成完成！已校准问题计数并应用居中对齐。")
    print(f" 文件保存位置: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description="Generate detailed Locomo Re-eval Excel Report")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing sample_conv json files")
    parser.add_argument("--output", type=str, default=None, help="Output Excel filename (default: saves to input_dir)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output if args.output else os.path.join(input_dir, f"locomo_reeval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    if args.output and not os.path.dirname(args.output):
        output_path = os.path.join(input_dir, args.output)

    
    data, meta = load_json_data(input_dir)
    
    if not data:
        print(f" 在 {input_dir} 未找到有效数据，请检查输入目录。")
        return

    create_excel_report(data, meta, output_path)

if __name__ == "__main__":
    main()