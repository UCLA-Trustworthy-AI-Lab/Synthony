import { Rows3, Columns3, AlertTriangle, CheckCircle } from 'lucide-react';
import { useDatasetStore } from '../../store';

export function ProfileSummary() {
    const { profile, columnAnalysis, activeDataset } = useDatasetStore();

    if (!profile || !activeDataset) {
        return null;
    }

    const stressFactors = profile.stress_factors;
    const activeStressFactors = Object.entries(stressFactors)
        .filter(([, value]) => value)
        .map(([key]) => key);

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">Profile Summary</span>
                <span className="badge badge-success">Analyzed</span>
            </div>
            <div className="card-body">
                {/* Dataset Info */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: '1rem',
                    marginBottom: '1.25rem',
                }}>
                    <div className="panel" style={{ textAlign: 'center', padding: '0.75rem' }}>
                        <Rows3 size={20} style={{ color: 'var(--text-muted)', marginBottom: '0.25rem' }} />
                        <div className="text-sm text-muted">Rows</div>
                        <div className="font-semibold">{profile.row_count.toLocaleString()}</div>
                    </div>
                    <div className="panel" style={{ textAlign: 'center', padding: '0.75rem' }}>
                        <Columns3 size={20} style={{ color: 'var(--text-muted)', marginBottom: '0.25rem' }} />
                        <div className="text-sm text-muted">Columns</div>
                        <div className="font-semibold">{profile.column_count}</div>
                    </div>
                    <div className="panel" style={{ textAlign: 'center', padding: '0.75rem' }}>
                        <div className="text-sm text-muted">Numeric</div>
                        <div className="font-semibold">{profile.numeric_columns}</div>
                    </div>
                    <div className="panel" style={{ textAlign: 'center', padding: '0.75rem' }}>
                        <div className="text-sm text-muted">Categorical</div>
                        <div className="font-semibold">{profile.categorical_columns}</div>
                    </div>
                </div>

                {/* Stress Factors */}
                <div style={{ marginBottom: '1rem' }}>
                    <div className="form-label" style={{ marginBottom: '0.5rem' }}>Stress Factors</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                        <StressBadge
                            label="Severe Skew"
                            active={stressFactors.severe_skew}
                        />
                        <StressBadge
                            label="High Cardinality"
                            active={stressFactors.high_cardinality}
                        />
                        <StressBadge
                            label="Zipfian Distribution"
                            active={stressFactors.zipfian_distribution}
                        />
                        <StressBadge
                            label="Small Data"
                            active={stressFactors.small_data}
                        />
                        <StressBadge
                            label="Large Data"
                            active={stressFactors.large_data}
                        />
                        <StressBadge
                            label="Higher-Order Correlation"
                            active={stressFactors.higher_order_correlation}
                        />
                    </div>
                </div>

                {/* Difficult Columns */}
                {columnAnalysis && columnAnalysis.difficult_columns.length > 0 && (
                    <div>
                        <div className="form-label" style={{ marginBottom: '0.5rem' }}>
                            Difficult Columns ({columnAnalysis.difficult_columns.length})
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                            {columnAnalysis.difficult_columns.map((col) => (
                                <span key={col} className="badge badge-warning">
                                    {col}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

                {/* Summary Message */}
                {activeStressFactors.length > 0 && (
                    <div style={{
                        marginTop: '1rem',
                        padding: '0.75rem',
                        background: 'rgba(245, 158, 11, 0.1)',
                        borderRadius: 'var(--border-radius)',
                        border: '1px solid rgba(245, 158, 11, 0.3)',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                            <AlertTriangle size={16} style={{ color: 'var(--warning)', marginTop: '2px' }} />
                            <div className="text-sm">
                                This dataset has <strong>{activeStressFactors.length}</strong> stress factor{activeStressFactors.length > 1 ? 's' : ''} that may affect synthesis quality.
                                Consider using models with high scores for these specific challenges.
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

interface StressBadgeProps {
    label: string;
    active: boolean;
}

function StressBadge({ label, active }: StressBadgeProps) {
    if (active) {
        return (
            <span className="badge badge-warning" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <AlertTriangle size={12} />
                {label}
            </span>
        );
    }

    return (
        <span className="badge badge-neutral" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', opacity: 0.6 }}>
            <CheckCircle size={12} />
            {label}
        </span>
    );
}
