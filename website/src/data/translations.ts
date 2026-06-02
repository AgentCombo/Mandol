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
  heroTitle_sub1: 'An In-Memory Layered Memory System',
  heroTitle_sub2: 'for Long-Term Conversational Agents',
  heroDescription:
    'Unified representation, efficient storage, and accurate retrieval of complex memory information — providing next-generation cognitive architecture for conversational agents.',
  heroPipInstall: 'pip install mandol',
  heroGetStarted: 'Get Started',
  heroViewOnGitHub: 'View on GitHub',
  highlightLoCoMo: 'LoCoMo Accuracy',
  highlightLongMemEval: 'LongMemEval Accuracy',
  highlightRetrievalSpeedup: 'Retrieval Speedup',
  highlightInsertionSpeedup: 'Insertion Speedup',

  whatIsTitle: 'What is Mandol?',
  whatIsDesc1:
    'Mandol is an in-memory, layered memory system for long-term conversational agents, with efficient and precise retrieval capabilities. It achieves unified representation, efficient storage, and accurate retrieval of complex memory information, providing theoretical foundations and technical solutions for next-generation agent cognitive architectures.',
  whatIsDesc2:
    'The system fuses key-value, vector, and graph indexing paradigms into a unified in-memory data structure, exposing a minimalist add() → holistic_retrieve() operational model. Its core innovation transforms the traditional "passive recall–rerank" retrieval paradigm into a proactive "Query-Aware Routing → quantitative denoising → high-quality context generation" paradigm.',

  innovationsTitle: 'Core Innovations',
  innovationsSubtitle: 'Three breakthroughs redefining how conversational agents remember',
  innovation1_title: 'Layered Memory Model',
  innovation1_desc:
    'A layered theoretical memory model dividing the system into base, high-level, and intelligent query layers, with a structured semantic graph unifying complex multi-relational memory representations.',
  innovation1_points: [
    'Structured semantic graph for unified representation of complex memory',
    'Implicit semantic edges generated on demand for precision–flexibility balance',
    'Bidirectional traceability linking base and high-level memories',
  ],
  innovation2_title: 'Unified In-Memory Storage',
  innovation2_desc:
    'A unified storage architecture based on in-memory semantic data structures, where co-designed SemanticMap and SemanticGraph fuse key-value, vector, and graph capabilities at the physical level.',
  innovation2_points: [
    'Native fusion of KV storage, vector indexing, and graph structures',
    'Atomic hybrid retrieval operators eliminate cross-store I/O bottlenecks',
    'Active-memory / durable-storage synergy balancing performance and capacity',
  ],
  innovation3_title: 'Intelligent Routing & Retrieval',
  innovation3_desc:
    'A Query-Aware Routing and quantitative retrieval method transforming retrieval from passive recall–rerank into a proactive understand–filter–summarize paradigm.',
  innovation3_points: [
    'Query-Aware Routing dynamically selects memory sources by intent',
    'Two-stage quantitative denoising with conflict resolution',
    'Token-constrained high-quality context generation without LLM in loop',
  ],

  benchmarkTitle: 'Benchmark Performance',
  benchmarkSubtitle: 'SOTA-level accuracy with lower token consumption on long-term conversational memory benchmarks',
  benchmarkLocomoTitle: 'LoCoMo Accuracy (%) Comparison',
  benchmarkLongmemTitle: 'LongMemEval Accuracy (%) Comparison',
  benchmarkLocomoNote:
    'Mandol achieves 92.21% overall on LoCoMo with only 1.9k tokens — outperforming EverMemOS (91.97% / 2.3k) while using 17% fewer tokens, and surpassing Mem0 (64.20%) by 28 points.',
  benchmarkLongmemNote:
    'Mandol achieves 88.40% overall on LongMemEval with 2.3k tokens — outperforming EverMemOS (83.00% / 2.8k) by 5.4 points while using 18% fewer tokens.',

  quickStartTitle: 'Quick Start',
  quickStartSubtitle: 'Three-step operational model: add → build → retrieve',
  quickStartTab1: 'Install',
  quickStartTab2: 'Insert Memory',
  quickStartTab3: 'Build & Retrieve',
  quickStartTab4: 'Save & Load',
  quickStartStep1: '1. Install',
  quickStartStep2: '2. Add Memories',
  quickStartStep3: '3. Build & Query',
  quickStartStep4: '4. Persist',
  quickStartCopy: 'Copy',
  quickStartCopied: 'Copied!',

  citationTitle: 'Citation',
  citationSubtitle: 'If this work is helpful to your research, please cite our paper',
  citationCopy: 'Copy BibTeX',
  citationCopied: 'Copied!',
  citationNote: 'The paper is forthcoming. The full author list and arXiv link will be updated upon publication.',

  footerDocs: 'Documentation',
  footerGitHub: 'GitHub',
  footerCommunity: 'Community',
  footerCopyright: 'Mandol Contributors. Apache 2.0 License.',
};

const zhHans: Translations = {
  heroBadgeDocs: '文档',
  heroTitle_sub1: '面向长期对话Agent的',
  heroTitle_sub2: '内存级分层记忆系统',
  heroDescription:
    '实现对复杂记忆信息的统一表示、高效存储和精确检索，为下一代 Agent 认知架构提供理论基础与技术方案。',
  heroPipInstall: 'pip install mandol',
  heroGetStarted: '快速开始',
  heroViewOnGitHub: 'GitHub 仓库',
  highlightLoCoMo: 'LoCoMo 正确率',
  highlightLongMemEval: 'LongMemEval 正确率',
  highlightRetrievalSpeedup: '检索加速',
  highlightInsertionSpeedup: '插入加速',

  whatIsTitle: '什么是 Mandol？',
  whatIsDesc1:
    'Mandol 是一个面向长期对话 Agent 的 内存级分层记忆系统，具备高效精准的检索能力。它实现了复杂记忆信息的统一表示、高效存储和精确检索，为下一代 Agent 认知架构提供了理论基础和技术方案。',
  whatIsDesc2:
    '系统将键值、向量和图索引范式融合为统一的内存数据结构，对外暴露极简的 add() → holistic_retrieve() 操作模型。其核心创新在于将传统的"被动召回-重排序"检索范式，转变为主动的"查询感知路由 → 量化去噪 → 高质量上下文生成"新范式。',

  innovationsTitle: '核心创新',
  innovationsSubtitle: '重新定义对话 Agent 记忆方式的三大突破',
  innovation1_title: '分层记忆模型',
  innovation1_desc:
    '将记忆系统划分为基础记忆层、高层记忆层和智能查询层的分层理论模型，通过结构化语义图统一表示复杂多关系记忆信息。',
  innovation1_points: [
    '结构化语义图统一表示复杂多关系记忆',
    '按需生成的隐式语义边实现精度与灵活性的平衡',
    '基础记忆与高层记忆的双向可追溯机制',
  ],
  innovation2_title: '统一内存存储架构',
  innovation2_desc:
    '基于内存语义数据结构的统一存储架构，SemanticMap 与 SemanticGraph 协同设计，在物理层面原生融合键值、向量和图能力。',
  innovation2_points: [
    'KV 存储、向量索引与图结构的原生融合',
    '原子化混合检索算子消除跨存储 I/O 瓶颈',
    '活跃内存-持久存储协同平衡性能与容量',
  ],
  innovation3_title: '智能路由与量化检索',
  innovation3_desc:
    '查询感知路由与量化检索方法，将检索过程从被动的"召回-重排序"转变为主动的"理解-过滤-总结"新范式。',
  innovation3_points: [
    '查询感知路由动态选择记忆来源',
    '两阶段量化去噪与冲突消解',
    'Token 约束下的高质量上下文生成，无需 LLM 参与检索',
  ],

  benchmarkTitle: '基准性能对比',
  benchmarkSubtitle: '在长期对话记忆基准上以更低 Token 消耗达到 SOTA 级正确率',
  benchmarkLocomoTitle: 'LoCoMo 正确率 (%) 对比',
  benchmarkLongmemTitle: 'LongMemEval 正确率 (%) 对比',
  benchmarkLocomoNote:
    'Mandol 在 LoCoMo 上以仅 1.9k tokens 达到 92.21% 的总体正确率 — 超越 EverMemOS (91.97% / 2.3k) 且 Token 消耗减少 17%，超越 Mem0 (64.20%) 达 28 个百分点。',
  benchmarkLongmemNote:
    'Mandol 在 LongMemEval 上以 2.3k tokens 达到 88.40% 的总体正确率 — 超越 EverMemOS (83.00% / 2.8k) 5.4 个百分点，同时 Token 消耗减少 18%。',

  quickStartTitle: '快速开始',
  quickStartSubtitle: '三步操作模型：add → build → retrieve',
  quickStartTab1: '安装',
  quickStartTab2: '插入记忆',
  quickStartTab3: '构建与检索',
  quickStartTab4: '保存与加载',
  quickStartStep1: '1. 安装',
  quickStartStep2: '2. 添加记忆',
  quickStartStep3: '3. 构建与查询',
  quickStartStep4: '4. 持久化',
  quickStartCopy: '复制',
  quickStartCopied: '已复制！',

  citationTitle: '引用',
  citationSubtitle: '如果您的研究受益于本工作，请引用我们的论文',
  citationCopy: '复制 BibTeX',
  citationCopied: '已复制！',
  citationNote: '论文即将发布。完整作者列表和 arXiv 链接将在出版后更新。',

  footerDocs: '文档',
  footerGitHub: 'GitHub',
  footerCommunity: '社区',
  footerCopyright: 'Mandol 贡献者。Apache 2.0 协议。',
};

export const translations: Record<Locale, Translations> = { en, 'zh-Hans': zhHans };
