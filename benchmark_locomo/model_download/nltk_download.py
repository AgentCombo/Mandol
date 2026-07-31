import nltk
import nltk
import logging

def download_nltk_resources():
    """Run download nltk resources."""
    resources = [
        'punkt',
        'punkt_tab',
        'wordnet',
        'omw-1.4',
        'stopwords'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logging.info(f" NLTK资源 {resource} 下载成功")
        except Exception as e:
            logging.warning(f" NLTK资源 {resource} 下载失败: {e}")

try:
    download_nltk_resources()
except Exception as e:
    logging.warning(f"NLTK资源下载过程中出现错误: {e}")

try:
    test_tokens = nltk.word_tokenize("This is a test.")
    logging.info(f"NLTK tokenize测试成功: {len(test_tokens)} tokens")
except Exception as e:
    logging.error(f"NLTK tokenize测试失败: {e}")

