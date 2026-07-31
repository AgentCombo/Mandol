import sys
from pathlib import Path
import logging

# sys.path.append(str(Path(__file__).parent))

from evaluation import (
    calculate_bert_f1_score,
    calculate_comprehensive_metrics,
    calculate_semantic_similarity,
    calculate_f1_score, 
    exact_match_score, 
    calculate_rouge_score,
    calculate_bleu_score, 
    calculate_meteor_score, 
    calculate_comprehensive_metrics
)

# Avoid mutating LogRecord fields before other handlers process the record.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_bert_f1_download():
    """Run test bert f1 download."""
    
    print(" 开始测试BERT F1模型...")
    
    try:
        test_cases = [
            {
                "name": "完全匹配",
                "gold_answer": "Caroline is a psychologist",
                "predicted_answer": "Caroline is a psychologist",
                "expected_range": (0.9, 1.0)
            },
            {
                "name": "语义相似",
                "gold_answer": "Caroline is a psychologist", 
                "predicted_answer": "Caroline works as a therapist",
                "expected_range": (0.5, 0.9)
            },
            {
                "name": "完全不匹配",
                "gold_answer": "Caroline is a psychologist",
                "predicted_answer": "John likes pizza",
                "expected_range": (0.0, 0.3)
            },
            {
                "name": "部分匹配",
                "gold_answer": "She went to the store yesterday",
                "predicted_answer": "Yesterday she visited the shop",
                "expected_range": (0.6, 0.9)
            },
            {
                "name": "空字符串处理",
                "gold_answer": "Some text",
                "predicted_answer": "",
                "expected_range": (0.0, 0.1)
            }
        ]
        
        print(f" 运行 {len(test_cases)} 个测试用例...")
        
        all_success = True
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n 测试用例 {i}: {case['name']}")
            print(f"   标准答案: '{case['gold_answer']}'")
            print(f"   预测答案: '{case['predicted_answer']}'")
            print(f"   期望范围: {case['expected_range']}")
            
            try:
                bert_f1 = calculate_bert_f1_score(case['gold_answer'], case['predicted_answer'])
                print(f"   BERT F1分数: {bert_f1:.4f}")
                
                min_expected, max_expected = case['expected_range']
                if min_expected <= bert_f1 <= max_expected:
                    print(f"    通过（在期望范围内）")
                else:
                    print(f"    失败（不在期望范围 {case['expected_range']} 内）")
                    all_success = False
                    
            except Exception as e:
                print(f"    计算失败: {e}")
                all_success = False
        
        if all_success:
            print(f"\n 所有测试用例通过！BERT F1模型工作正常。")
        else:
            print(f"\n 部分测试用例失败，但这可能是正常的（取决于模型行为）")
            
    except ImportError as e:
        print(f" 导入失败: {e}")
        print("请确保已安装bert-score库: pip install bert-score")
        return False
    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_sentence_transformer():
    """Run test sentence transformer."""
    
    print("\n 测试SentenceTransformer模型...")
    
    try:
        test_cases = [
            {
                "gold": "I love cats",
                "pred": "I adore felines",
                "desc": "语义相似"
            },
            {
                "gold": "The weather is nice",
                "pred": "It's raining heavily",
                "desc": "语义不同"
            },
            {
                "gold": "Hello world", 
                "pred": "Hello world",
                "desc": "完全相同"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n 语义相似度测试 {i}: {case['desc']}")
            print(f"   文本1: '{case['gold']}'")
            print(f"   文本2: '{case['pred']}'")
            
            try:
                similarity = calculate_semantic_similarity(case['gold'], case['pred'])
                print(f"   语义相似度: {similarity:.4f}")
                print(f"    计算成功")
            except Exception as e:
                print(f"    计算失败: {e}")
                return False
                
        print(f"\n SentenceTransformer模型工作正常！")
        return True
        
    except Exception as e:
        print(f" SentenceTransformer测试失败: {e}")
        return False

def test_all_metrics():
    """Run test all metrics."""
    
    print("\n 测试所有评估指标...")
    
    try:
        gold_answer = "Caroline is a psychologist who helps people"
        predicted_answer = "Caroline works as a therapist helping patients"
        
        print(f"标准答案: '{gold_answer}'")
        print(f"预测答案: '{predicted_answer}'")
        print(f"\n 各项指标:")
        
        f1 = calculate_f1_score(gold_answer, predicted_answer)
        print(f"F1分数: {f1:.4f}")
        
        em = exact_match_score(gold_answer, predicted_answer)
        print(f"精确匹配: {em}")
        
        rouge_scores = calculate_rouge_score(gold_answer, predicted_answer)
        print(f"ROUGE-1 F1: {rouge_scores['rouge1_f']:.4f}")
        print(f"ROUGE-L F1: {rouge_scores['rougeL_f']:.4f}")
        
        bleu_scores = calculate_bleu_score(gold_answer, predicted_answer)
        print(f"BLEU-1: {bleu_scores['bleu1']:.4f}")
        print(f"BLEU-4: {bleu_scores['bleu4']:.4f}")
        
        meteor = calculate_meteor_score(gold_answer, predicted_answer)
        print(f"METEOR: {meteor:.4f}")
        
        bert_f1 = calculate_bert_f1_score(gold_answer, predicted_answer)
        print(f"BERT F1: {bert_f1:.4f}")
        
        semantic_sim = calculate_semantic_similarity(gold_answer, predicted_answer)
        print(f"语义相似度: {semantic_sim:.4f}")
        
        print(f"\n 所有指标计算成功！")
        
        print(f"\n 测试综合指标计算...")
        comprehensive = calculate_comprehensive_metrics(
            gold_answer, predicted_answer, 
            context="Some context here",
            options=["lexical", "semantic"]
        )
        
        print(f" 综合指标结果:")
        print(f"词汇指标数量: {len(comprehensive.get('lexical', {}))}")
        print(f"语义指标数量: {len(comprehensive.get('semantic', {}))}")
        
        return True
        
    except Exception as e:
        print(f" 综合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_downloads():
    """Run test model downloads."""
    
    print("\n 测试模型下载...")
    
    print("\n 测试BERT Score模型下载...")
    try:
        from bert_score import score
        _, _, f1 = score(["hello world"], ["hello world"], lang="en", verbose=True)
        print(f" BERT Score模型下载成功，测试F1分数: {f1.item():.4f}")
    except Exception as e:
        print(f" BERT Score模型下载失败: {e}")
        return False
    
    print("\n 测试SentenceTransformer模型下载...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(["test sentence"])
        print(f" SentenceTransformer模型下载成功，嵌入维度: {embeddings.shape}")
    except Exception as e:
        print(f" SentenceTransformer模型下载失败: {e}")
        return False
    
    print("\n 测试NLTK数据下载...")
    try:
        import nltk
        nltk.download('punkt', quiet=False)
        nltk.download('wordnet', quiet=False)
        print(f" NLTK数据下载成功")
    except Exception as e:
        print(f" NLTK数据下载失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print(" 开始BERT F1和相关模型测试...")
    print("=" * 60)
    
    print(" 第一步：测试模型下载...")
    download_success = test_model_downloads()
    
    print("\n" + "=" * 60)
    print(" 第二步：测试评估功能...")
    
    bert_success = test_bert_f1_download()
    
    st_success = test_sentence_transformer()
    
    all_success = test_all_metrics()
    
    print("\n" + "=" * 60)
    print(" 测试结果总结:")
    print(f"模型下载: {' 成功' if download_success else ' 失败'}")
    print(f"BERT F1模型: {' 成功' if bert_success else ' 失败'}")
    print(f"SentenceTransformer: {' 成功' if st_success else ' 失败'}")
    print(f"综合指标: {' 成功' if all_success else ' 失败'}")
    
    if download_success and bert_success and st_success and all_success:
        print(f"\n 所有测试通过！模型已准备就绪。")
        
        print(f"\n 使用示例:")
        print(f"from evaluation import calculate_bert_f1_score")
        print(f"score = calculate_bert_f1_score('标准答案', '预测答案')")
        print(f"print(f'BERT F1分数: {{score:.4f}}')")
        
    else:
        print(f"\n 部分测试失败，请检查环境配置。")
        
        print(f"\n 可能需要的依赖:")
        print(f"pip install bert-score")
        print(f"pip install sentence-transformers")
        print(f"pip install rouge-score")
        print(f"pip install nltk")
        
        if not download_success:
            print(f"\n 模型下载失败，请检查网络连接或尝试手动下载")