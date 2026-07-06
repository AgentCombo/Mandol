import os
import json
import pandas as pd
import glob
import argparse
from datetime import datetime
import numpy as np

CATEGORY_NAMES = {
    1: "多跳问题(Multi-hop)",
    2: "时间问题(Temporal)",
    3: "开放域问题(Open-domain)",
    4: "单跳问题(Single-hop)",
    5: "对抗性问题(Adversarial)",
}

def load_json_data(input_dir):
    """Load json data."""
    root_path = os.path.abspath(input_dir)
    direct_files = glob.glob(os.path.join(root_path, "*.json"))
    recursive_files = glob.glob(os.path.join(root_path, "**", "*.json"), recursive=True)
    files = sorted(set(direct_files + recursive_files))
    
    all_details = []
    sample_meta_map = {}  # Dataset-specific handling used by the reproduction workflow.
    
    print(f" 正在扫描目录: {input_dir}")
    if not direct_files and recursive_files:
        print("ℹ  输入目录本身没有 JSON，已递归扫描子实验目录")

    def get_experiment_name(file_path: str) -> str:
        parent = os.path.dirname(os.path.abspath(file_path))
        rel_parent = os.path.relpath(parent, root_path)
        if rel_parent == ".":
            return os.path.basename(root_path.rstrip(os.sep))
        return rel_parent.replace(os.sep, "/")

    def make_sample_key(experiment_name: str, sample_id: str) -> str:
        if not sample_id or sample_id == "unknown":
            return sample_id
        
        if experiment_name == os.path.basename(root_path.rstrip(os.sep)):
            return sample_id
        return f"{experiment_name}/{sample_id}"
    
    summary_files = [f for f in files if "final_summary" in os.path.basename(f)]
    if summary_files:
        print(f" 发现 {len(summary_files)} 个汇总文件，将用于校准统计数据")
    for summary_file in summary_files:
        try:
            experiment_name = get_experiment_name(summary_file)
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                if "sample_summaries" in summary_data:
                    for sid, s_info in summary_data["sample_summaries"].items():
                        count = s_info.get("successful_count", s_info.get("test_count", 0))
                        sample_meta_map[make_sample_key(experiment_name, sid)] = count
        except Exception as e:
            print(f" 读取汇总文件 {summary_file} 失败: {e}")

    valid_files_count = 0
    for file_path in files:
        fname = os.path.basename(file_path).lower()
        # Dataset-specific handling used by the reproduction workflow.
        if "final_summary" in fname or "readable" in fname or "report" in fname:
            if "sample" not in fname: continue

        try:
            experiment_name = get_experiment_name(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                details = []
                # Dataset-specific handling used by the reproduction workflow.
                file_sample_id = "unknown"
                if isinstance(data, dict):
                    # Dataset-specific handling used by the reproduction workflow.
                    s_info = data.get('sample_info', {})
                    file_sample_id = s_info.get('sample_id', 'unknown')
                    sample_key = make_sample_key(experiment_name, file_sample_id)
                    
                    if sample_key not in sample_meta_map and file_sample_id != "unknown":
                        sample_meta_map[sample_key] = s_info.get('successful_count', len(details))

                    if 'results' in data: details = data['results']
                    elif 'detailed_results' in data: details = data['detailed_results']
                    elif 'qa_results' in data: details = data['qa_results']
                    elif 'aggregate_results' in data: details = data['aggregate_results']
                elif isinstance(data, list):
                    details = data
                    sample_key = make_sample_key(experiment_name, file_sample_id)
                
                if not details: continue
                valid_files_count += 1

                for item in details:
                    item['_source_file'] = fname
                    item['experiment'] = experiment_name
                    if 'sample_id' not in item:
                        item['sample_id'] = sample_key
                    
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
            print(f"  读取文件 {file_path} 失败: {e}")
            
    print(f" 已处理 {valid_files_count} 个详细结果文件")
    return all_details, sample_meta_map

def create_excel_report(details_data, sample_meta_map, output_file):
    if not details_data:
        print(" 无有效数据")
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

    
    
    df_no_adv = df[df['category'] != CATEGORY_NAMES.get(5, '对抗性问题(Adversarial)')]
    
    if sample_meta_map:
        total_questions = sum(sample_meta_map.values())
    else:
        total_questions = len(df)

    adv_count_in_df = len(df[df['category'] == CATEGORY_NAMES.get(5, '对抗性问题(Adversarial)')])
    non_adv_total = total_questions - adv_count_in_df

    dashboard_data = [
        {'Metric (指标)': 'Avg Input Tokens (平均输入Token)', 'Value': df['tokens_input'].mean()},
        {'Metric (指标)': 'Avg Retrieval Time (平均检索时间 s)', 'Value': df['total_retrieval_time'].mean()},
        {'Metric (指标)': '--------------------------------', 'Value': ''},
        {'Metric (指标)': 'Avg LLM Accuracy [All] (平均LLM准确率-含对抗)',
         'Value': df['score_llm_accuracy'].fillna(0).sum() / total_questions if total_questions > 0 else 0},
        {'Metric (指标)': 'Avg LLM Accuracy [No Adv] (平均LLM准确率-除对抗)',
         'Value': df_no_adv['score_llm_accuracy'].fillna(0).sum() / non_adv_total if non_adv_total > 0 else 0},
        {'Metric (指标)': '--------------------------------', 'Value': ''},
        {'Metric (指标)': 'Avg F1-Score (平均F1分数)', 'Value': df['score_token_f1'].mean()},
        {'Metric (指标)': 'Total Questions (总问题数 - 基于元数据)', 'Value': total_questions},
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
    if 'sample_id' in df.columns:
        def calc_acc(x):
            sid = x.name # group key
            real_count = sample_meta_map.get(sid, len(x))
            adv_rows_in_sample = len(x[x['category'] == CATEGORY_NAMES.get(5, '对抗性问题(Adversarial)')])
            non_adv_count = real_count - adv_rows_in_sample
            non_adv_data = x[x['category'] != CATEGORY_NAMES.get(5, '对抗性问题(Adversarial)')]
            
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
    else:
        df_sample_acc = pd.DataFrame()

    
    
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
    
    if 'sample_id' in df.columns:
        df_time_sample = df.groupby('sample_id').agg(agg_time).reset_index().rename(columns={'question': 'Computed Count'})
    if 'category' in df.columns:
        df_time_cat = df.groupby('category').agg(agg_time).reset_index().rename(columns={'question': 'Computed Count'})

    
    # Sheet 5: Token_Stats
    
    target_tokens = [c for c in token_cols if c in df.columns]
    agg_token = {c: 'mean' for c in target_tokens}
    agg_token['question'] = 'count'
    
    df_token_sample = pd.DataFrame()
    df_token_cat = pd.DataFrame()
    
    if 'sample_id' in df.columns:
        df_token_sample = df.groupby('sample_id').agg(agg_token).reset_index().rename(columns={'question': 'Computed Count'})
    if 'category' in df.columns:
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

    df_dashboard.to_excel(writer, sheet_name='Dashboard', index=False)
    ws = writer.sheets['Dashboard']
    ws.set_column('A:A', 45, text_left_fmt)
    ws.set_column('B:B', 20, num_fmt)
    for col_num, value in enumerate(df_dashboard.columns.values):
        ws.write(0, col_num, value, header_fmt)
    ws.write(4, 1, df_dashboard.iloc[3]['Value'], percent_fmt)  # Avg LLM Accuracy [All]
    ws.write(5, 1, df_dashboard.iloc[4]['Value'], percent_fmt)  # Avg LLM Accuracy [No Adv]
    if not df_sample_acc.empty:
        df_sample_acc.to_excel(writer, sheet_name='Sample_LLM_Stats', index=False)
        ws = writer.sheets['Sample_LLM_Stats']
        ws.set_column('A:A', 15, text_center_fmt)  # Dataset-specific handling used by the reproduction workflow.
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

    if not df_cat_acc.empty:
        df_cat_acc.to_excel(writer, sheet_name='Category_Accuracy', index=False)
        ws = writer.sheets['Category_Accuracy']
        ws.set_column('A:A', 28, text_left_fmt)
        ws.set_column('B:B', 12, int_fmt)         # Count
        ws.set_column('C:F', 20, percent_fmt)
        for col_num, value in enumerate(df_cat_acc.columns.values):
            ws.write(0, col_num, value, header_fmt)
        ws.conditional_format('C2:F100', {'type': '3_color_scale'})

    ws = workbook.add_worksheet('Time_Stats')
    writer.sheets['Time_Stats'] = ws
    row = 0
    if not df_time_sample.empty:
        ws.write(row, 0, "按样本平均耗时 (Average Time per Sample) [Seconds]", title_fmt)
        row += 1
        for col_num, val in enumerate(df_time_sample.columns): ws.write(row, col_num, val, header_fmt)
        for r_idx, r_val in enumerate(df_time_sample.values):
            ws.write(row+1+r_idx, 0, r_val[0], text_center_fmt) # ID
            for c_idx, cell_val in enumerate(r_val[1:], 1): 
                fmt = int_fmt if "Count" in df_time_sample.columns[c_idx] else num_fmt
                ws.write(row+1+r_idx, c_idx, cell_val, fmt)
        row += len(df_time_sample) + 3

    if not df_time_cat.empty:
        ws.write(row, 0, "按类别平均耗时 (Average Time per Category) [Seconds]", title_fmt)
        row += 1
        for col_num, val in enumerate(df_time_cat.columns): ws.write(row, col_num, val, header_fmt)
        for r_idx, r_val in enumerate(df_time_cat.values):
            ws.write(row+1+r_idx, 0, r_val[0], text_center_fmt)
            for c_idx, cell_val in enumerate(r_val[1:], 1):
                fmt = int_fmt if "Count" in df_time_cat.columns[c_idx] else num_fmt
                ws.write(row+1+r_idx, c_idx, cell_val, fmt)
    ws.set_column(0, 0, 15)
    ws.set_column(1, 20, 18)

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
            ws.write(row+1+r_idx, 0, r_val[0], text_center_fmt)
            for c_idx, cell_val in enumerate(r_val[1:], 1): ws.write(row+1+r_idx, c_idx, cell_val, int_fmt)
    ws.set_column(0, 0, 15)
    ws.set_column(1, 20, 18)

    df_details.to_excel(writer, sheet_name='Detailed_Logs', index=False)
    ws = writer.sheets['Detailed_Logs']
    
    for col_num, value in enumerate(df_details.columns.values):
        ws.write(0, col_num, value, header_fmt)

    ws.set_column('A:A', 15, text_center_fmt) # Sample ID
    ws.set_column('B:B', 10, text_center_fmt)  # Category
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
    print(" Excel 报告生成完成！已校准问题计数并应用居中对齐。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=".")
    parser.add_argument('--output-file', type=str, default=None)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_path = args.output_file if args.output_file else os.path.join(input_dir, f"locomo_report_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    data, meta = load_json_data(input_dir)
    create_excel_report(data, meta, output_path)

if __name__ == "__main__":
    main()
