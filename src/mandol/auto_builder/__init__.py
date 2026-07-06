# mandol/auto_builder/__init__.py
"""Package exports for auto builder."""

from .session_tracker import SessionTracker, SessionInfo
from .auto_session_assigner import (
    AutoSessionAssigner,
    AutoSessionConfig,
    ActiveSessionState,
    SessionAssignment,
)
from .hierarchical_builder import (
    HierarchicalAutoBuilder,
    HierarchicalBuilderConfig,
    L1ExtractionResult,
    L2AggregationResult
)
from .hierarchical_prompts import (
    HierarchicalPromptManager,
    HierarchicalPromptTemplateManager,  
    ExtractionStyle,
    L1SummaryType
)
from .entity_relation_builder import (
    EntityRelationAutoBuilder, 
    EntityRelationBuilderConfig,
    ExtractedEntity, 
    ExtractedRelation,
    EntityMention,
    MergedEntity
)
from .entity_relation_prompts import EntityRelationPromptManager, EntityType
from .episodic_builder import EpisodicAutoBuilder, EpisodicBuilderConfig, EpisodicFact, TimeInfo
from .episodic_prompts import EpisodicPromptTemplateManager, EpisodicFactType
from .graph_write_queue import GraphWriteQueue, GraphWriteRequest, dispatch_graph_write_requests
from .strategy_config import PipelineStrategy, STYLE_STRATEGIES
from .orchestrator import MemoryOrchestrator, OrchestratorConfig, PipelineResult
from .high_level_memory import (
    HighLevelMemoryBuilder,
    HighLevelMemoryBuildConfig,
    HighLevelMemoryBuildResult,
    build_high_level_memory,
)

__version__ = "1.4.0"

__all__ = [
    "SessionTracker",
    "SessionInfo",
    "AutoSessionAssigner",
    "AutoSessionConfig",
    "ActiveSessionState",
    "SessionAssignment",
    
    "HierarchicalAutoBuilder",
    "HierarchicalBuilderConfig",
    "HierarchicalPromptManager",
    "HierarchicalPromptTemplateManager",  
    "ExtractionStyle",
    "L1SummaryType",
    "L1ExtractionResult",
    "L2AggregationResult",
    
    "EntityRelationAutoBuilder",
    "EntityRelationBuilderConfig",
    "EntityRelationPromptManager",
    "EntityType",
    "ExtractedEntity",
    "ExtractedRelation",
    "EntityMention",
    "MergedEntity",
    
    "EpisodicAutoBuilder",
    "EpisodicBuilderConfig",
    "EpisodicPromptTemplateManager",
    "EpisodicFactType",
    "EpisodicFact",
    "TimeInfo",

    "GraphWriteQueue",
    "GraphWriteRequest",
    "dispatch_graph_write_requests",

    "PipelineStrategy",
    "STYLE_STRATEGIES",
    
    "MemoryOrchestrator",
    "OrchestratorConfig",
    "PipelineResult",
    "HighLevelMemoryBuilder",
    "HighLevelMemoryBuildConfig",
    "HighLevelMemoryBuildResult",
    "build_high_level_memory",
]
