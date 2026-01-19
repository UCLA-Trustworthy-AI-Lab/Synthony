import { Calculator, MessageSquare, Layers } from 'lucide-react';
import { useModeStore } from '../../store';
import type { RecommendationMode } from '../../types';

const modes: { id: RecommendationMode; label: string; description: string; icon: typeof Calculator }[] = [
    {
        id: 'rule-based',
        label: 'Rule-Based',
        description: 'Fast deterministic scoring',
        icon: Calculator,
    },
    {
        id: 'llm',
        label: 'LLM',
        description: 'AI-powered recommendations',
        icon: MessageSquare,
    },
    {
        id: 'hybrid',
        label: 'Hybrid',
        description: 'Best of both approaches',
        icon: Layers,
    },
];

export function ModeSelector() {
    const { mode, setMode } = useModeStore();

    return (
        <div className="mode-selector">
            <div className="card-title" style={{ marginBottom: '0.5rem', paddingLeft: '0.25rem' }}>
                Recommendation Mode
            </div>
            {modes.map((m) => {
                const Icon = m.icon;
                return (
                    <button
                        key={m.id}
                        className={`mode-option ${mode === m.id ? 'active' : ''}`}
                        onClick={() => setMode(m.id)}
                    >
                        <Icon className="mode-icon" size={20} />
                        <div style={{ flex: 1, textAlign: 'left' }}>
                            <div className="mode-label">{m.label}</div>
                            <div className="mode-description">{m.description}</div>
                        </div>
                    </button>
                );
            })}
        </div>
    );
}
