import { useState, useRef, useEffect } from 'react';
import { Send, Settings, RefreshCw, Paperclip } from 'lucide-react';
import { useChatStore, useDatasetStore } from '../../store';
import { v4 as uuidv4 } from 'uuid';
import type { ChatMessage } from '../../types';

export function ChatInterface() {
    const { messages, addMessage, isLoading, setLoading, llmConfig, setLLMConfig } = useChatStore();
    const { activeDataset, profile } = useDatasetStore();
    const [input, setInput] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: ChatMessage = {
            id: uuidv4(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date(),
        };

        addMessage(userMessage);
        setInput('');
        setLoading(true);

        // Simulate LLM response (replace with actual API call)
        setTimeout(() => {
            const assistantMessage: ChatMessage = {
                id: uuidv4(),
                role: 'assistant',
                content: generateMockResponse(input, activeDataset?.filename, profile),
                timestamp: new Date(),
                metadata: {
                    model: llmConfig.model,
                },
            };
            addMessage(assistantMessage);
            setLoading(false);
        }, 1500);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const quickPrompts = [
        'Analyze my data',
        'Compare top 3 models',
        'Explain trade-offs',
        'What handles skew best?',
    ];

    return (
        <div className="chat-container" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Model Configuration Bar */}
            <div className="panel" style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                padding: '0.75rem 1rem',
                borderBottom: '1px solid var(--border-default)',
                borderRadius: '0',
            }}>
                <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
                    <label className="form-label" style={{ margin: 0, whiteSpace: 'nowrap' }}>Model:</label>
                    <select
                        className="form-input form-select"
                        value={llmConfig.model}
                        onChange={(e) => setLLMConfig({ model: e.target.value })}
                        style={{ width: '160px' }}
                    >
                        <option value="gpt-4-turbo">GPT-4 Turbo</option>
                        <option value="gpt-4">GPT-4</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                    </select>
                </div>

                <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
                    <label className="form-label" style={{ margin: 0 }}>Temp:</label>
                    <input
                        type="number"
                        className="form-input"
                        value={llmConfig.temperature}
                        onChange={(e) => setLLMConfig({ temperature: parseFloat(e.target.value) })}
                        min={0}
                        max={2}
                        step={0.1}
                        style={{ width: '70px' }}
                    />
                </div>

                <div style={{ flex: 1 }} />

                <button
                    className="btn btn-ghost btn-icon"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    title="Advanced Options"
                >
                    <Settings size={18} />
                </button>

                <button
                    className="btn btn-ghost btn-icon"
                    onClick={() => useChatStore.getState().clearMessages()}
                    title="New Conversation"
                >
                    <RefreshCw size={18} />
                </button>
            </div>

            {/* Advanced Options Panel */}
            {showAdvanced && (
                <div className="panel" style={{
                    padding: '1rem',
                    borderBottom: '1px solid var(--border-default)',
                    borderRadius: '0',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: '1rem',
                }}>
                    <div className="form-group">
                        <label className="form-label">Max Tokens</label>
                        <input
                            type="number"
                            className="form-input"
                            value={llmConfig.max_tokens}
                            onChange={(e) => setLLMConfig({ max_tokens: parseInt(e.target.value) })}
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Top P</label>
                        <input
                            type="number"
                            className="form-input"
                            value={llmConfig.top_p || 0.95}
                            onChange={(e) => setLLMConfig({ top_p: parseFloat(e.target.value) })}
                            min={0}
                            max={1}
                            step={0.05}
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Frequency Penalty</label>
                        <input
                            type="number"
                            className="form-input"
                            value={llmConfig.frequency_penalty || 0}
                            onChange={(e) => setLLMConfig({ frequency_penalty: parseFloat(e.target.value) })}
                            min={-2}
                            max={2}
                            step={0.1}
                        />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Presence Penalty</label>
                        <input
                            type="number"
                            className="form-input"
                            value={llmConfig.presence_penalty || 0}
                            onChange={(e) => setLLMConfig({ presence_penalty: parseFloat(e.target.value) })}
                            min={-2}
                            max={2}
                            step={0.1}
                        />
                    </div>
                </div>
            )}

            {/* Chat Messages */}
            <div className="chat-messages" style={{ flex: 1, overflowY: 'auto' }}>
                {messages.length === 0 ? (
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        color: 'var(--text-muted)',
                    }}>
                        <div style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>💬 Start a conversation</div>
                        <div style={{ fontSize: '0.875rem' }}>
                            {activeDataset
                                ? `Ask about recommendations for ${activeDataset.filename}`
                                : 'Upload a dataset to get started'}
                        </div>
                    </div>
                ) : (
                    messages.map((message) => (
                        <div key={message.id} className={`message ${message.role}`}>
                            <div className="message-avatar">
                                {message.role === 'user' ? '👤' : '🤖'}
                            </div>
                            <div className="message-content">
                                <div className="message-text" style={{ whiteSpace: 'pre-wrap' }}>
                                    {message.content}
                                </div>
                            </div>
                        </div>
                    ))
                )}

                {isLoading && (
                    <div className="message assistant">
                        <div className="message-avatar">🤖</div>
                        <div className="message-content">
                            <div className="typing-indicator">
                                <div className="typing-dot" />
                                <div className="typing-dot" />
                                <div className="typing-dot" />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Quick Prompts */}
            {messages.length === 0 && (
                <div style={{
                    padding: '0.75rem 1rem',
                    display: 'flex',
                    gap: '0.5rem',
                    flexWrap: 'wrap',
                    borderTop: '1px solid var(--border-default)',
                }}>
                    {quickPrompts.map((prompt) => (
                        <button
                            key={prompt}
                            className="btn btn-secondary btn-sm"
                            onClick={() => setInput(prompt)}
                        >
                            {prompt}
                        </button>
                    ))}
                </div>
            )}

            {/* Input Area */}
            <div className="chat-input-container">
                <div className="chat-input-wrapper">
                    <button className="btn btn-ghost btn-icon" title="Attach context">
                        <Paperclip size={18} />
                    </button>
                    <textarea
                        className="chat-input"
                        placeholder="Type your message..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={1}
                    />
                    <button
                        className="btn btn-primary"
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
}

// Mock response generator (replace with actual API integration)
function generateMockResponse(
    input: string,
    filename?: string,
    profile?: { stress_factors?: { severe_skew?: boolean; high_cardinality?: boolean } } | null
): string {
    const lowerInput = input.toLowerCase();

    if (lowerInput.includes('skew') || lowerInput.includes('skewness')) {
        return `Based on your question about handling skewed data, I recommend **GReaT** as the primary choice.

**Why GReaT?**
• Score: 4/4 for extreme skewness handling (>4.0 skew)
• Uses LLM-based generation that naturally captures distribution tails
• Superior rare category capture

**Alternatives:**
1. **TabDDPM** (Score: 3.8) - Strong diffusion-based approach
2. **TabTree** (Score: 3.5) - Tree-based, handles moderate skew well

⚠️ **Note**: GReaT has slower training time (hours for 10k+ rows).`;
    }

    if (lowerInput.includes('compare') || lowerInput.includes('top')) {
        return `Here's a comparison of the top 3 recommended models${filename ? ` for ${filename}` : ''}:

| Model | Score | Strengths | Weaknesses |
|-------|-------|-----------|------------|
| **GReaT** | 4.2 | Skew, Cardinality | Slow training |
| **TabDDPM** | 3.8 | Correlations, Quality | Needs GPU |
| **ARF** | 3.5 | Fast, Small data | Limited scalability |

Would you like me to explain any of these in more detail?`;
    }

    if (lowerInput.includes('trade') || lowerInput.includes('explain')) {
        return `**Key Trade-offs to Consider:**

🚀 **Speed vs Quality**
- Fast models (ARF, SMOTE): Minutes to train, but lower fidelity
- High-quality models (GReaT, TabDDPM): Hours to train, excellent results

🔒 **Privacy vs Utility**
- DP-enabled models provide formal privacy guarantees
- But adding DP noise reduces synthetic data quality

💻 **Hardware Requirements**
- CPU-only: GReaT, ARF, CTGAN, SMOTE
- GPU required: TabDDPM, TabSyn, STaSy

What aspect would you like to explore further?`;
    }

    return `Thank you for your question${filename ? ` about ${filename}` : ''}!

${profile?.stress_factors?.severe_skew ? '⚠️ I notice your dataset has **severe skewness**. ' : ''}${profile?.stress_factors?.high_cardinality ? '⚠️ Your data also shows **high cardinality**. ' : ''}

Based on my analysis, I recommend exploring:
1. **GReaT** - Best overall for complex data patterns
2. **TabDDPM** - Excellent for preserving correlations
3. **ARF** - Great for smaller datasets

Would you like me to provide more specific recommendations based on your constraints?`;
}
