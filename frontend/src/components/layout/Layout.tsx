import { Play, RotateCcw } from 'lucide-react';
import { useModeStore, useDatasetStore, useConstraintsStore } from '../../store';
import { ModeSelector } from '../mode/ModeSelector';
import { UploadZone } from '../data/UploadZone';
import { ProfileSummary } from '../data/ProfileSummary';
import { ConstraintForm } from '../rules/ConstraintForm';
import { ChatInterface } from '../chat/ChatInterface';
import { ResultsPanel } from '../results/ResultsPanel';
import { apiClient } from '../../api/client';

export function Layout() {
    const { mode } = useModeStore();
    const { activeDataset, profile, columnAnalysis, datasets } = useDatasetStore();
    const { constraints, topN, setRecommendation, setRecommending, isRecommending } = useConstraintsStore();

    const handleRunAnalysis = async () => {
        if (!activeDataset || !profile || !columnAnalysis) return;

        setRecommending(true);
        try {
            const result = await apiClient.getRecommendation(
                activeDataset.dataset_id,
                profile,
                columnAnalysis,
                constraints,
                mode,
                topN
            );
            setRecommendation(result);
        } catch (error) {
            console.error('Recommendation failed:', error);
        } finally {
            setRecommending(false);
        }
    };

    return (
        <div className="app-layout">
            {/* Left Sidebar */}
            <aside className="sidebar">
                {/* Logo/Header */}
                <div style={{
                    padding: '1rem 1.25rem',
                    borderBottom: '1px solid var(--border-default)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                }}>
                    <div style={{
                        width: '32px',
                        height: '32px',
                        background: 'linear-gradient(135deg, var(--primary-500), var(--primary-700))',
                        borderRadius: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '1rem',
                    }}>
                        🧪
                    </div>
                    <div>
                        <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>Synthony</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Model Recommender</div>
                    </div>
                </div>

                {/* Mode Selection */}
                <ModeSelector />

                {/* Dataset List */}
                <div style={{ padding: '1rem', borderTop: '1px solid var(--border-default)', flex: 1 }}>
                    <div className="card-title" style={{ marginBottom: '0.75rem' }}>Datasets</div>
                    {datasets.length === 0 ? (
                        <div className="text-sm text-muted">No datasets uploaded</div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {datasets.map((ds) => (
                                <div
                                    key={ds.dataset_id}
                                    className={`panel ${ds.dataset_id === activeDataset?.dataset_id ? 'active' : ''}`}
                                    style={{
                                        padding: '0.75rem',
                                        cursor: 'pointer',
                                        border: ds.dataset_id === activeDataset?.dataset_id
                                            ? '1px solid var(--primary-500)'
                                            : '1px solid transparent',
                                    }}
                                    onClick={() => useDatasetStore.getState().setActiveDataset(ds)}
                                >
                                    <div className="text-sm font-medium truncate">{ds.filename}</div>
                                    <div className="text-xs text-muted">
                                        {ds.rows.toLocaleString()} rows • {ds.columns} cols
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {/* Header */}
                <header className="header">
                    <h1 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                        {mode === 'rule-based' && 'Rule-Based Recommendation'}
                        {mode === 'llm' && 'LLM-Powered Recommendation'}
                        {mode === 'hybrid' && 'Hybrid Recommendation'}
                    </h1>
                    <div style={{ flex: 1 }} />
                    {mode !== 'llm' && activeDataset && profile && (
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <button
                                className="btn btn-secondary"
                                onClick={() => useConstraintsStore.getState().resetConstraints()}
                            >
                                <RotateCcw size={16} />
                                Reset
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={handleRunAnalysis}
                                disabled={isRecommending}
                            >
                                <Play size={16} />
                                {isRecommending ? 'Analyzing...' : 'Run Analysis'}
                            </button>
                        </div>
                    )}
                </header>

                {/* Content Area */}
                <div className="content-area">
                    {/* Top Row: Upload Zone */}
                    {!activeDataset && (
                        <UploadZone />
                    )}

                    {/* Mode-specific content */}
                    {mode === 'rule-based' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                <ConstraintForm />
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                {activeDataset && <ProfileSummary />}
                                <ResultsPanel />
                            </div>
                        </div>
                    )}

                    {mode === 'llm' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '1.5rem', flex: 1 }}>
                            <div className="card" style={{ display: 'flex', flexDirection: 'column', minHeight: '500px' }}>
                                <ChatInterface />
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                {activeDataset && <ProfileSummary />}
                                <ResultsPanel />
                            </div>
                        </div>
                    )}

                    {mode === 'hybrid' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr 350px', gap: '1.5rem', flex: 1 }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                <ConstraintForm />
                            </div>
                            <div className="card" style={{ display: 'flex', flexDirection: 'column', minHeight: '400px' }}>
                                <ChatInterface />
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                {activeDataset && <ProfileSummary />}
                                <ResultsPanel />
                            </div>
                        </div>
                    )}

                    {/* Data Preview (always shown if dataset is active) */}
                    {activeDataset && (
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">Data Preview: {activeDataset.filename}</span>
                                <span className="text-sm text-muted">
                                    {activeDataset.rows.toLocaleString()} rows × {activeDataset.columns} columns
                                </span>
                            </div>
                            <div className="card-body" style={{ padding: 0, overflow: 'auto' }}>
                                {activeDataset.preview && activeDataset.preview.length > 0 ? (
                                    <table className="data-table">
                                        <thead>
                                            <tr>
                                                {Object.keys(activeDataset.preview[0]).map((key) => (
                                                    <th key={key}>{key}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {activeDataset.preview.slice(0, 5).map((row, i) => (
                                                <tr key={i}>
                                                    {Object.values(row).map((value, j) => (
                                                        <td key={j}>{String(value)}</td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                ) : (
                                    <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                                        No preview available
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
