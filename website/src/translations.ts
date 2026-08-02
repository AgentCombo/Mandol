export type Locale = 'en' | 'zh-Hans';

export interface Translations {
  // Hero
  heroBadgeDocs: string;
  heroTitle_sub1: string;
  heroTitle_sub2: string;
  heroDescription: string;
  heroPipInstall: string;
  heroGetStarted: string;
  heroViewOnGitHub: string;
  // Highlights row
  highlightLoCoMo: string;
  highlightLongMemEval: string;
  highlightRetrievalSpeedup: string;
  highlightInsertionSpeedup: string;

  // What is Mandol
  whatIsTitle: string;
  whatIsDesc1: string;
  whatIsDesc2: string;

  // Core Innovations
  innovationsTitle: string;
  innovationsSubtitle: string;
  innovation1_title: string;
  innovation1_desc: string;
  innovation1_points: string[];
  innovation2_title: string;
  innovation2_desc: string;
  innovation2_points: string[];
  innovation3_title: string;
  innovation3_desc: string;
  innovation3_points: string[];

  // Benchmark
  benchmarkTitle: string;
  benchmarkSubtitle: string;
  benchmarkLocomoTitle: string;
  benchmarkLongmemTitle: string;
  benchmarkLocomoNote: string;
  benchmarkLongmemNote: string;
  benchmarkLatencyTitle: string;
  benchmarkLatencySubtitle: string;
  benchmarkServerTitle: string;
  benchmarkServerEnvironment: string;
  benchmarkMetricSearch: string;
  benchmarkMetricAdd: string;
  benchmarkMetricRetrievalSpeedup: string;
  benchmarkMetricInsertionSpeedup: string;
  benchmarkMetricServerContext: string;
  benchmarkMetricBaselineContext: string;
  benchmarkSystem: string;
  benchmarkSearch: string;
  benchmarkAdd: string;
  benchmarkMean: string;
  benchmarkOurs: string;
  benchmarkLocalEyebrow: string;
  benchmarkLocalTitle: string;
  benchmarkLocalDescription: string;
  benchmarkLocalSeparate: string;
  benchmarkLatencyConditions: string;
  benchmarkLatencyEvermemNote: string;
  benchmarkLatencySourcePrefix: string;
  benchmarkLatencySourceLink: string;
  benchmarkLatencySourceSuffix: string;

  // Quick Start
  quickStartTitle: string;
  quickStartSubtitle: string;
  quickStartTab1: string;
  quickStartTab2: string;
  quickStartTab3: string;
  quickStartTab4: string;
  quickStartStep1: string;
  quickStartStep2: string;
  quickStartStep3: string;
  quickStartStep4: string;
  quickStartCopy: string;
  quickStartCopied: string;

  // Citation
  citationTitle: string;
  citationSubtitle: string;
  citationCopy: string;
  citationCopied: string;
  citationNote: string;

  // Footer
  footerDocs: string;
  footerGitHub: string;
  footerCommunity: string;
  footerCopyright: string;
}

const en: Translations = {
  heroBadgeDocs: 'Documentation',
  heroTitle_sub1: 'An In-Process Semantic Memory Runtime',
  heroTitle_sub2: 'for Agent Retrieval Systems',
  heroDescription:
    'Current Mandol exposes MemoryUnit, SemanticMap, SemanticGraph, MultiRetriever, and three-tower retrieval components from the src/mandol package.',
  heroPipInstall: 'python -m pip install mandol',
  heroGetStarted: 'Get Started',
  heroViewOnGitHub: 'View on GitHub',
  highlightLoCoMo: 'Python Runtime',
  highlightLongMemEval: 'SemanticMap',
  highlightRetrievalSpeedup: 'SemanticGraph',
  highlightInsertionSpeedup: 'MultiRetriever',

  whatIsTitle: 'What is Mandol?',
  whatIsDesc1:
    'Mandol is an in-process semantic memory package for agent retrieval experiments. Its indexes and graph topology remain resident, with optional RocksDB-backed paging for cold MemoryUnit payloads.',
  whatIsDesc2:
    'Retrieval is handled through MultiRetriever for BM25, SPLADE, cosine search, graph expansion, score fusion, and reranker orchestration. Current development is on main; exact reproduction of the published experiments uses the frozen paper-repro branch.',

  innovationsTitle: 'Current Architecture',
  innovationsSubtitle: 'The maintained runtime is built from explicit, inspectable memory primitives',
  innovation1_title: 'Core Memory Objects',
  innovation1_desc:
    'MemoryUnit stores the payload, MemorySpace stores logical membership, and MemorySpaceRegistry defines canonical three-tower spaces.',
  innovation1_points: [
    'String UIDs and dictionary raw_data payloads',
    'Tree-shaped spaces backed by UID references',
    'Canonical hierarchical, entity-relation, and episodic tower names',
  ],
  innovation2_title: 'SemanticMap + SemanticGraph',
  innovation2_desc:
    'SemanticMap owns embeddings, FAISS indexing, sparse vectors and space-filtered search; SemanticGraph adds rustworkx relationships and graph persistence.',
  innovation2_points: [
    'Global FAISS index with MemorySpace filtering',
    'BM25 and SPLADE index integration through retrieval modules',
    'Complete graph snapshots with RocksDB-backed tiered payload paging',
  ],
  innovation3_title: 'Retrieval Orchestration',
  innovation3_desc:
    'MultiRetriever and TripleTowerRetriever provide retrieval-facing orchestration for dense, sparse, graph, episodic and entity-relation paths.',
  innovation3_points: [
    'BM25, SPLADE and cosine retrieval with RRF fusion',
    'Optional graph expansion and reranker management',
    'Async paths for vLLM-backed rerankers',
  ],

  benchmarkTitle: 'Benchmark Performance',
  benchmarkSubtitle: 'SOTA-level accuracy with lower token consumption on long-term conversational memory benchmarks',
  benchmarkLocomoTitle: 'LoCoMo Accuracy (%) Comparison',
  benchmarkLongmemTitle: 'LongMemEval Accuracy (%) Comparison',
  benchmarkLocomoNote:
    'Mandol achieves 92.21% overall on LoCoMo with only 1.9k tokens. Paper rows use generated three-tower graphs with router + quantification; Qwen/DeepSeek build and deduplicate memories, while GPT-4.1-mini/GPT-4o-mini are used for task evaluation and judging.',
  benchmarkLongmemNote:
    'Mandol achieves 88.40% overall on LongMemEval with 2.3k tokens. Reproduction requires the cleaned dataset plus generated hierarchical, episodic, and entity-relation graph artifacts before running task_eval.',
  benchmarkLatencyTitle: 'Retrieval and Insertion Latency',
  benchmarkLatencySubtitle:
    'Complete tail and mean latency results, with server and local deployments reported separately',
  benchmarkServerTitle: 'Server Deployment',
  benchmarkServerEnvironment: 'NVIDIA H800 80GB · Search 10 QPS · Add 10 QPS',
  benchmarkMetricSearch: 'Mean Search Latency',
  benchmarkMetricAdd: 'Mean Add Latency',
  benchmarkMetricRetrievalSpeedup: 'Retrieval Speedup',
  benchmarkMetricInsertionSpeedup: 'Insertion Speedup',
  benchmarkMetricServerContext: 'H800 · 10 QPS',
  benchmarkMetricBaselineContext: 'server mean · vs. best non-Mandol baseline',
  benchmarkSystem: 'System',
  benchmarkSearch: 'Search',
  benchmarkAdd: 'Add',
  benchmarkMean: 'Mean',
  benchmarkOurs: 'Ours',
  benchmarkLocalEyebrow: 'Local deployment',
  benchmarkLocalTitle: 'Consumer Hardware',
  benchmarkLocalDescription:
    'On an NVIDIA RTX 5090 Laptop 24GB, Mandol records 166.5 ms mean search latency at 5 QPS and 37.4 ms mean insertion latency at 10 QPS.',
  benchmarkLocalSeparate:
    'This result is shown separately because the hardware and search load differ from the server comparison.',
  benchmarkLatencyConditions:
    'All values are milliseconds. Server measurements use an NVIDIA H800 80GB at 10 QPS for Search and Add. Local measurements use an NVIDIA RTX 5090 Laptop 24GB at 5 QPS for Search and 10 QPS for Add.',
  benchmarkLatencyEvermemNote: '† EverMemOS was reproduced using its official implementation.',
  benchmarkLatencySourcePrefix: 'These results correspond to the frozen',
  benchmarkLatencySourceLink: 'paper reproduction artifact',
  benchmarkLatencySourceSuffix: '.',

  quickStartTitle: 'Quick Start',
  quickStartSubtitle: 'Install the package or source environment, add MemoryUnit records, search with MultiRetriever, then persist the graph',
  quickStartTab1: 'Environment',
  quickStartTab2: 'Add Units',
  quickStartTab3: 'Search',
  quickStartTab4: 'Persist',
  quickStartStep1: '1. Environment',
  quickStartStep2: '2. Units',
  quickStartStep3: '3. Search',
  quickStartStep4: '4. Persist',
  quickStartCopy: 'Copy',
  quickStartCopied: 'Copied!',

  citationTitle: 'Citation',
  citationSubtitle: 'If this work is helpful to your research, please cite our paper',
  citationCopy: 'Copy BibTeX',
  citationCopied: 'Copied!',
  citationNote: 'The paper is available as arXiv:2606.29778.',

  footerDocs: 'Documentation',
  footerGitHub: 'GitHub',
  footerCommunity: 'Community',
  footerCopyright: 'Mandol Contributors. Apache 2.0 License.',
};

const zhHans: Translations = {
  heroBadgeDocs: '文档',
  heroTitle_sub1: '面向智能体检索系统的',
  heroTitle_sub2: '进程内语义记忆运行时',
  heroDescription:
    '当前 Mandol 从 src/mandol 暴露 MemoryUnit、SemanticMap、SemanticGraph、MultiRetriever 与三塔检索组件。',
  heroPipInstall: 'python -m pip install mandol',
  heroGetStarted: '快速开始',
  heroViewOnGitHub: 'GitHub 仓库',
  highlightLoCoMo: 'Python 运行时',
  highlightLongMemEval: 'SemanticMap',
  highlightRetrievalSpeedup: 'SemanticGraph',
  highlightInsertionSpeedup: 'MultiRetriever',

  whatIsTitle: '什么是 Mandol？',
  whatIsDesc1:
    'Mandol 是一个面向智能体检索实验的进程内语义记忆包。索引与图拓扑保持常驻，冷 MemoryUnit payload 可选择通过 RocksDB-backed 换页管理。',
  whatIsDesc2:
    '检索由 MultiRetriever 负责，覆盖 BM25、SPLADE、余弦搜索、图扩展、分数融合和 reranker 编排。当前开发位于 main；精确复现已发表实验时应使用冻结的 paper-repro 分支。',

  innovationsTitle: '当前架构',
  innovationsSubtitle: '维护中的运行时由明确、可检查的记忆原语组成',
  innovation1_title: '核心记忆对象',
  innovation1_desc:
    'MemoryUnit 存储载荷，MemorySpace 存储逻辑归属，MemorySpaceRegistry 定义三塔检索使用的规范空间。',
  innovation1_points: [
    '字符串 UID 与字典 raw_data 载荷',
    '基于 UID 引用的树形空间',
    '分层、实体关系、情景三类规范空间名',
  ],
  innovation2_title: 'SemanticMap + SemanticGraph',
  innovation2_desc:
    'SemanticMap 管理 embedding、FAISS、稀疏向量和空间过滤搜索；SemanticGraph 增加 rustworkx 关系图和图级持久化。',
  innovation2_points: [
    '全局 FAISS 索引与 MemorySpace 过滤',
    '通过 retrieval 模块集成 BM25 与 SPLADE 索引',
    '完整图快照与 RocksDB-backed 自动 payload 换页',
  ],
  innovation3_title: '检索编排',
  innovation3_desc:
    'MultiRetriever 与 TripleTowerRetriever 为稠密、稀疏、图、情景和实体关系路径提供检索编排。',
  innovation3_points: [
    'BM25、SPLADE、余弦检索与 RRF 融合',
    '可选图扩展与 reranker 管理',
    '面向 vLLM reranker 的异步路径',
  ],

  benchmarkTitle: '基准性能对比',
  benchmarkSubtitle: '在长期对话记忆基准上以更低 Token 消耗达到 SOTA 级正确率',
  benchmarkLocomoTitle: 'LoCoMo 正确率 (%) 对比',
  benchmarkLongmemTitle: 'LongMemEval 正确率 (%) 对比',
  benchmarkLocomoNote:
    'Mandol 在 LoCoMo 上以 1.9k tokens 达到 92.21% 的总体正确率。论文表格使用生成好的三塔图，并启用 router + quantification；Qwen/DeepSeek 用于记忆生成和去重，GPT-4.1-mini/GPT-4o-mini 用于 task-eval 与 judge。',
  benchmarkLongmemNote:
    'Mandol 在 LongMemEval 上以 2.3k tokens 达到 88.40% 的总体正确率。复现时需要先准备 cleaned 数据集，并生成 hierarchical、episodic 与 entity-relation 三类图产物后再运行 task_eval。',
  benchmarkLatencyTitle: '检索与记忆写入延迟',
  benchmarkLatencySubtitle: '完整展示尾延迟与平均延迟，服务器和本地部署分别报告',
  benchmarkServerTitle: '服务器部署',
  benchmarkServerEnvironment: 'NVIDIA H800 80GB · 检索 10 QPS · 写入 10 QPS',
  benchmarkMetricSearch: '平均检索延迟',
  benchmarkMetricAdd: '平均写入延迟',
  benchmarkMetricRetrievalSpeedup: '检索加速',
  benchmarkMetricInsertionSpeedup: '记忆写入加速',
  benchmarkMetricServerContext: 'H800 · 10 QPS',
  benchmarkMetricBaselineContext: '服务器平均延迟 · 相比最佳非 Mandol 基线',
  benchmarkSystem: '系统',
  benchmarkSearch: '检索',
  benchmarkAdd: '写入',
  benchmarkMean: '平均值',
  benchmarkOurs: '本系统',
  benchmarkLocalEyebrow: '本地部署',
  benchmarkLocalTitle: '消费级硬件',
  benchmarkLocalDescription:
    '在 NVIDIA RTX 5090 Laptop 24GB 上，Mandol 在 5 QPS 检索负载下的平均延迟为 166.5 ms，在 10 QPS 写入负载下的平均延迟为 37.4 ms。',
  benchmarkLocalSeparate: '由于硬件和检索负载与服务器横向对比不同，该结果单独展示，不参与服务器系统排名。',
  benchmarkLatencyConditions:
    '所有数值单位均为毫秒。服务器实验使用 NVIDIA H800 80GB，检索和写入均为 10 QPS；本地实验使用 NVIDIA RTX 5090 Laptop 24GB，检索为 5 QPS，写入为 10 QPS。',
  benchmarkLatencyEvermemNote: '† EverMemOS 使用其官方实现复现。',
  benchmarkLatencySourcePrefix: '以上数据对应冻结的',
  benchmarkLatencySourceLink: '论文复现 artifact',
  benchmarkLatencySourceSuffix: '。',

  quickStartTitle: '快速开始',
  quickStartSubtitle: '安装发布包或源码环境，添加 MemoryUnit，用 MultiRetriever 检索，再持久化图快照',
  quickStartTab1: '环境',
  quickStartTab2: '添加单元',
  quickStartTab3: '检索',
  quickStartTab4: '持久化',
  quickStartStep1: '1. 环境',
  quickStartStep2: '2. 单元',
  quickStartStep3: '3. 检索',
  quickStartStep4: '4. 持久化',
  quickStartCopy: '复制',
  quickStartCopied: '已复制！',

  citationTitle: '引用',
  citationSubtitle: '如果您的研究受益于本工作，请引用我们的论文',
  citationCopy: '复制 BibTeX',
  citationCopied: '已复制！',
  citationNote: '论文已发布于 arXiv:2606.29778。',

  footerDocs: '文档',
  footerGitHub: 'GitHub',
  footerCommunity: '社区',
  footerCopyright: 'Mandol 贡献者。Apache 2.0 协议。',
};

export const translations: Record<Locale, Translations> = { en, 'zh-Hans': zhHans };
