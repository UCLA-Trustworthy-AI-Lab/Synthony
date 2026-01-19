import type {
    DatasetInfo,
    AnalysisResponse,
    RecommendationResult,
    HealthResponse,
    Constraints,
    RecommendationMode
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    private async request<T>(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`;
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    // Health check
    async getHealth(): Promise<HealthResponse> {
        return this.request<HealthResponse>('/health');
    }

    // Upload dataset
    async uploadDataset(
        file: File,
        sessionId?: string,
        datasetName?: string
    ): Promise<DatasetInfo> {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams();
        if (sessionId) params.append('session_id', sessionId);
        if (datasetName) params.append('dataset_name', datasetName);

        const queryString = params.toString();
        const endpoint = `/api/v1/dataloader/upload${queryString ? `?${queryString}` : ''}`;

        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    // Analyze dataset
    async analyzeDataset(file: File, datasetId?: string): Promise<AnalysisResponse> {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams();
        if (datasetId) params.append('dataset_id', datasetId);

        const queryString = params.toString();
        const endpoint = `/analyze${queryString ? `?${queryString}` : ''}`;

        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Analysis failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    // Get recommendations
    async getRecommendation(
        datasetId: string,
        profile: unknown,
        columnAnalysis: unknown,
        constraints: Constraints,
        method: RecommendationMode,
        topN: number = 3
    ): Promise<RecommendationResult> {
        return this.request<RecommendationResult>('/recommend', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                dataset_profile: profile,
                column_analysis: columnAnalysis,
                constraints,
                method: method === 'rule-based' ? 'rule_based' : method,
                top_n: topN,
            }),
        });
    }

    // Analyze and recommend in one call
    async analyzeAndRecommend(
        file: File,
        options: {
            datasetId?: string;
            method?: RecommendationMode;
            cpuOnly?: boolean;
            strictDp?: boolean;
            topN?: number;
        } = {}
    ): Promise<{ dataset_id: string; analysis: AnalysisResponse; recommendation: RecommendationResult }> {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams();
        if (options.datasetId) params.append('dataset_id', options.datasetId);
        if (options.method) params.append('method', options.method === 'rule-based' ? 'rule_based' : options.method);
        if (options.cpuOnly !== undefined) params.append('cpu_only', String(options.cpuOnly));
        if (options.strictDp !== undefined) params.append('strict_dp', String(options.strictDp));
        if (options.topN !== undefined) params.append('top_n', String(options.topN));

        const queryString = params.toString();
        const endpoint = `/analyze-and-recommend${queryString ? `?${queryString}` : ''}`;

        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    // List models
    async listModels(filters?: {
        modelType?: string;
        cpuOnly?: boolean;
        requiresDp?: boolean;
    }): Promise<{ total_models: number; models: Record<string, unknown> }> {
        const params = new URLSearchParams();
        if (filters?.modelType) params.append('model_type', filters.modelType);
        if (filters?.cpuOnly !== undefined) params.append('cpu_only', String(filters.cpuOnly));
        if (filters?.requiresDp !== undefined) params.append('requires_dp', String(filters.requiresDp));

        const queryString = params.toString();
        return this.request(`/models${queryString ? `?${queryString}` : ''}`);
    }

    // Get model details
    async getModel(modelName: string): Promise<Record<string, unknown>> {
        return this.request(`/models/${modelName}`);
    }
}

export const apiClient = new ApiClient(API_BASE);
