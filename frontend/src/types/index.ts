// Store types
export type RecommendationMode = 'rule-based' | 'llm' | 'hybrid';

export interface AppState {
    mode: RecommendationMode;
    setMode: (mode: RecommendationMode) => void;
}

// Dataset types
export interface DatasetInfo {
    dataset_id: string;
    session_id: string;
    filename: string;
    format: 'csv' | 'parquet';
    rows: number;
    columns: number;
    size_bytes: number;
    created_at: string;
    status: 'uploading' | 'uploaded' | 'analyzing' | 'analyzed' | 'error';
    preview?: Record<string, unknown>[];
    analyze_url?: string;
}

export interface StressFactors {
    severe_skew: boolean;
    high_cardinality: boolean;
    zipfian_distribution: boolean;
    small_data: boolean;
    large_data: boolean;
    higher_order_correlation: boolean;
}

export interface DatasetProfile {
    row_count: number;
    column_count: number;
    numeric_columns: number;
    categorical_columns: number;
    stress_factors: StressFactors;
    skewness_metrics?: Record<string, number>;
    cardinality_metrics?: Record<string, number>;
    zipfian_metrics?: Record<string, number>;
}

export interface ColumnAnalysis {
    max_column_difficulty: number;
    difficult_columns: string[];
    columns: Record<string, unknown>;
}

// Recommendation types
export interface Constraints {
    cpu_only: boolean;
    strict_dp: boolean;
    dp_epsilon?: number;
    priority_weights?: {
        speed: number;
        quality: number;
        privacy: number;
    };
}

export interface ModelCapabilities {
    skew_handling: number;
    cardinality_handling: number;
    zipfian_handling: number;
    small_data: number;
    correlation_handling: number;
    privacy_dp: number;
}

export interface RecommendedModel {
    model_name: string;
    confidence_score: number;
    capability_match: ModelCapabilities;
    reasoning: string[];
    warnings: string[];
    model_info?: Record<string, unknown>;
}

export interface RecommendationResult {
    dataset_id: string;
    method: RecommendationMode;
    recommended_model: RecommendedModel;
    alternative_models: RecommendedModel[];
    llm_reasoning?: string;
    constraints: Constraints;
    difficulty_summary?: Record<string, unknown>;
    excluded_models?: Record<string, string>;
}

// Chat types
export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    metadata?: {
        model?: string;
        tokens?: number;
        recommendations?: RecommendedModel[];
    };
}

export interface LLMConfig {
    provider: 'openai' | 'anthropic' | 'local';
    model: string;
    temperature: number;
    max_tokens: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
}

// API Response types
export interface AnalysisResponse {
    dataset_id: string;
    dataset_profile: DatasetProfile;
    column_analysis: ColumnAnalysis;
    message: string;
}

export interface HealthResponse {
    status: string;
    version: string;
    analyzer_available: boolean;
    recommender_available: boolean;
    llm_available: boolean;
    models_count: number;
}
