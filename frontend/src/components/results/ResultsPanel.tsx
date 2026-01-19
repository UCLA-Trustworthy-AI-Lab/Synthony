import { Star, AlertTriangle, CheckCircle } from 'lucide-react';
import { useConstraintsStore } from '../../store';
import type { RecommendedModel } from '../../types';

export function ResultsPanel() {
    const { recommendation, isRecommending } = useConstraintsStore();

    if (isRecommending) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Recommendation Results</span>
                </div>
                <div className="card-body" style={{ textAlign: 'center', padding: '3rem' }}>
                    <div className="animate-pulse" style={{ color: 'var(--text-muted)' }}>
                        Analyzing and generating recommendations...
                    </div>
                </div>
            </div>
        );
    }

    if (!recommendation) {
        return (
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Recommendation Results</span>
                </div>
                <div className="card-body" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    Upload a dataset and run analysis to get recommendations
                </div>
            </div>
        );
    }

    const allModels = [recommendation.recommended_model, ...recommendation.alternative_models];

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">Recommendation Results</span>
                <span className="badge badge-info">{recommendation.method.replace('_', '-')}</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {allModels.map((model, index) => (
                    <ModelResultCard
                        key={model.model_name}
                        model={model}
                        rank={index + 1}
                        isTop={index === 0}
                    />
                ))}

                {recommendation.excluded_models && Object.keys(recommendation.excluded_models).length > 0 && (
                    <div style={{ marginTop: '1rem' }}>
                        <div className="text-sm text-muted" style={{ marginBottom: '0.5rem' }}>
                            Excluded Models:
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                            {Object.entries(recommendation.excluded_models).map(([name, reason]) => (
                                <span key={name} className="badge badge-neutral" title={reason}>
                                    {name}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

interface ModelResultCardProps {
    model: RecommendedModel;
    rank: number;
    isTop: boolean;
}

function ModelResultCard({ model, rank, isTop }: ModelResultCardProps) {
    const percentage = Math.round(model.confidence_score * 100);

    return (
        <div className={`result-card ${isTop ? 'recommended' : ''}`}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                <div className="result-rank">#{rank}</div>

                <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                        <span className="result-model-name">{model.model_name}</span>
                        {isTop && <Star size={16} style={{ color: 'var(--warning)', fill: 'var(--warning)' }} />}
                        <span className="result-score">Score: {model.confidence_score.toFixed(2)}</span>
                    </div>

                    {/* Score Bar */}
                    <div className="score-bar" style={{ marginBottom: '0.75rem' }}>
                        <div
                            className="score-bar-fill"
                            style={{ width: `${percentage}%` }}
                        />
                    </div>

                    {/* Reasoning */}
                    {model.reasoning.length > 0 && (
                        <div style={{ marginBottom: '0.5rem' }}>
                            {model.reasoning.slice(0, 2).map((reason, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.25rem' }}>
                                    <CheckCircle size={14} style={{ color: 'var(--success)', marginTop: '2px', flexShrink: 0 }} />
                                    <span className="text-sm">{reason}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Warnings */}
                    {model.warnings.length > 0 && (
                        <div>
                            {model.warnings.slice(0, 1).map((warning, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                                    <AlertTriangle size={14} style={{ color: 'var(--warning)', marginTop: '2px', flexShrink: 0 }} />
                                    <span className="text-sm text-muted">{warning}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="text-sm font-semibold" style={{ color: isTop ? 'var(--primary-400)' : 'var(--text-muted)' }}>
                    {percentage}%
                </div>
            </div>
        </div>
    );
}
