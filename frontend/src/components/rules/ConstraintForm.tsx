import { useConstraintsStore } from '../../store';

export function ConstraintForm() {
    const { constraints, setConstraints, topN, setTopN, resetConstraints } = useConstraintsStore();

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">System Constraints</span>
                <button className="btn btn-ghost btn-sm" onClick={resetConstraints}>
                    Reset
                </button>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {/* Hardware Requirements */}
                <div>
                    <div className="form-label" style={{ marginBottom: '0.75rem' }}>Hardware Requirements</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        <label className="checkbox-group">
                            <input
                                type="checkbox"
                                className="checkbox"
                                checked={constraints.cpu_only}
                                onChange={(e) => setConstraints({ cpu_only: e.target.checked })}
                            />
                            <span className="checkbox-label">CPU Only (No GPU)</span>
                        </label>
                    </div>
                </div>

                {/* Privacy & Compliance */}
                <div>
                    <div className="form-label" style={{ marginBottom: '0.75rem' }}>Privacy & Compliance</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        <label className="checkbox-group">
                            <input
                                type="checkbox"
                                className="checkbox"
                                checked={constraints.strict_dp}
                                onChange={(e) => setConstraints({ strict_dp: e.target.checked })}
                            />
                            <span className="checkbox-label">Strict DP (ε &lt; 1)</span>
                        </label>

                        {constraints.strict_dp && (
                            <div className="form-group" style={{ marginTop: '0.5rem', marginLeft: '1.875rem' }}>
                                <label className="form-label">DP Budget (ε)</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={constraints.dp_epsilon || 1.0}
                                    onChange={(e) => setConstraints({ dp_epsilon: parseFloat(e.target.value) })}
                                    min={0.1}
                                    max={10}
                                    step={0.1}
                                    style={{ width: '100px' }}
                                />
                            </div>
                        )}
                    </div>
                </div>

                {/* Priority Sliders */}
                <div>
                    <div className="form-label" style={{ marginBottom: '0.75rem' }}>Priority Weighting</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div className="slider-container">
                            <div className="slider-header">
                                <span className="slider-label">Speed</span>
                                <span className="slider-value">{Math.round((constraints.priority_weights?.speed || 0.5) * 100)}%</span>
                            </div>
                            <input
                                type="range"
                                className="slider"
                                min={0}
                                max={1}
                                step={0.1}
                                value={constraints.priority_weights?.speed || 0.5}
                                onChange={(e) => setConstraints({
                                    priority_weights: {
                                        ...constraints.priority_weights,
                                        speed: parseFloat(e.target.value),
                                        quality: constraints.priority_weights?.quality || 0.5,
                                        privacy: constraints.priority_weights?.privacy || 0.5,
                                    }
                                })}
                            />
                        </div>

                        <div className="slider-container">
                            <div className="slider-header">
                                <span className="slider-label">Quality</span>
                                <span className="slider-value">{Math.round((constraints.priority_weights?.quality || 0.5) * 100)}%</span>
                            </div>
                            <input
                                type="range"
                                className="slider"
                                min={0}
                                max={1}
                                step={0.1}
                                value={constraints.priority_weights?.quality || 0.5}
                                onChange={(e) => setConstraints({
                                    priority_weights: {
                                        ...constraints.priority_weights,
                                        speed: constraints.priority_weights?.speed || 0.5,
                                        quality: parseFloat(e.target.value),
                                        privacy: constraints.priority_weights?.privacy || 0.5,
                                    }
                                })}
                            />
                        </div>

                        <div className="slider-container">
                            <div className="slider-header">
                                <span className="slider-label">Privacy</span>
                                <span className="slider-value">{Math.round((constraints.priority_weights?.privacy || 0.5) * 100)}%</span>
                            </div>
                            <input
                                type="range"
                                className="slider"
                                min={0}
                                max={1}
                                step={0.1}
                                value={constraints.priority_weights?.privacy || 0.5}
                                onChange={(e) => setConstraints({
                                    priority_weights: {
                                        ...constraints.priority_weights,
                                        speed: constraints.priority_weights?.speed || 0.5,
                                        quality: constraints.priority_weights?.quality || 0.5,
                                        privacy: parseFloat(e.target.value),
                                    }
                                })}
                            />
                        </div>
                    </div>
                </div>

                {/* Output Options */}
                <div>
                    <div className="form-label" style={{ marginBottom: '0.75rem' }}>Output Options</div>
                    <div className="form-group">
                        <label className="form-label">Top N Results</label>
                        <select
                            className="form-input form-select"
                            value={topN}
                            onChange={(e) => setTopN(parseInt(e.target.value))}
                            style={{ width: '100px' }}
                        >
                            <option value={1}>1</option>
                            <option value={3}>3</option>
                            <option value={5}>5</option>
                            <option value={10}>10</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    );
}
