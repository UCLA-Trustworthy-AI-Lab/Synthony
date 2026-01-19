import { create } from 'zustand';
import type {
    RecommendationMode,
    DatasetInfo,
    DatasetProfile,
    ColumnAnalysis,
    ChatMessage,
    LLMConfig,
    Constraints,
    RecommendationResult
} from '../types';

// Mode Store
interface ModeState {
    mode: RecommendationMode;
    setMode: (mode: RecommendationMode) => void;
}

export const useModeStore = create<ModeState>((set) => ({
    mode: 'hybrid',
    setMode: (mode) => set({ mode }),
}));

// Dataset Store
interface DatasetState {
    datasets: DatasetInfo[];
    activeDataset: DatasetInfo | null;
    profile: DatasetProfile | null;
    columnAnalysis: ColumnAnalysis | null;
    isUploading: boolean;
    isAnalyzing: boolean;
    error: string | null;

    addDataset: (dataset: DatasetInfo) => void;
    setActiveDataset: (dataset: DatasetInfo | null) => void;
    setProfile: (profile: DatasetProfile | null) => void;
    setColumnAnalysis: (analysis: ColumnAnalysis | null) => void;
    setUploading: (uploading: boolean) => void;
    setAnalyzing: (analyzing: boolean) => void;
    setError: (error: string | null) => void;
    updateDatasetStatus: (id: string, status: DatasetInfo['status']) => void;
}

export const useDatasetStore = create<DatasetState>((set) => ({
    datasets: [],
    activeDataset: null,
    profile: null,
    columnAnalysis: null,
    isUploading: false,
    isAnalyzing: false,
    error: null,

    addDataset: (dataset) => set((state) => ({
        datasets: [...state.datasets, dataset],
        activeDataset: dataset,
    })),
    setActiveDataset: (dataset) => set({ activeDataset: dataset }),
    setProfile: (profile) => set({ profile }),
    setColumnAnalysis: (analysis) => set({ columnAnalysis: analysis }),
    setUploading: (uploading) => set({ isUploading: uploading }),
    setAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
    setError: (error) => set({ error }),
    updateDatasetStatus: (id, status) => set((state) => ({
        datasets: state.datasets.map((d) =>
            d.dataset_id === id ? { ...d, status } : d
        ),
        activeDataset: state.activeDataset?.dataset_id === id
            ? { ...state.activeDataset, status }
            : state.activeDataset,
    })),
}));

// Chat Store
interface ChatState {
    messages: ChatMessage[];
    isLoading: boolean;
    llmConfig: LLMConfig;

    addMessage: (message: ChatMessage) => void;
    updateMessage: (id: string, content: string) => void;
    clearMessages: () => void;
    setLoading: (loading: boolean) => void;
    setLLMConfig: (config: Partial<LLMConfig>) => void;
}

export const useChatStore = create<ChatState>((set) => ({
    messages: [],
    isLoading: false,
    llmConfig: {
        provider: 'openai',
        model: 'gpt-4-turbo',
        temperature: 0.7,
        max_tokens: 2048,
    },

    addMessage: (message) => set((state) => ({
        messages: [...state.messages, message],
    })),
    updateMessage: (id, content) => set((state) => ({
        messages: state.messages.map((m) =>
            m.id === id ? { ...m, content: m.content + content } : m
        ),
    })),
    clearMessages: () => set({ messages: [] }),
    setLoading: (loading) => set({ isLoading: loading }),
    setLLMConfig: (config) => set((state) => ({
        llmConfig: { ...state.llmConfig, ...config },
    })),
}));

// Constraints Store
interface ConstraintsState {
    constraints: Constraints;
    topN: number;
    recommendation: RecommendationResult | null;
    isRecommending: boolean;

    setConstraints: (constraints: Partial<Constraints>) => void;
    setTopN: (n: number) => void;
    setRecommendation: (result: RecommendationResult | null) => void;
    setRecommending: (recommending: boolean) => void;
    resetConstraints: () => void;
}

const defaultConstraints: Constraints = {
    cpu_only: false,
    strict_dp: false,
    priority_weights: {
        speed: 0.5,
        quality: 0.5,
        privacy: 0.5,
    },
};

export const useConstraintsStore = create<ConstraintsState>((set) => ({
    constraints: defaultConstraints,
    topN: 3,
    recommendation: null,
    isRecommending: false,

    setConstraints: (constraints) => set((state) => ({
        constraints: { ...state.constraints, ...constraints },
    })),
    setTopN: (topN) => set({ topN }),
    setRecommendation: (recommendation) => set({ recommendation }),
    setRecommending: (recommending) => set({ isRecommending: recommending }),
    resetConstraints: () => set({ constraints: defaultConstraints, topN: 3 }),
}));
