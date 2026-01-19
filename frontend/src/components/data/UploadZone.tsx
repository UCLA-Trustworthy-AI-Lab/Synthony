import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileSpreadsheet, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { useDatasetStore } from '../../store';
import { apiClient } from '../../api/client';

export function UploadZone() {
    const {
        isUploading,
        setUploading,
        addDataset,
        setProfile,
        setColumnAnalysis,
        setError,
        setAnalyzing
    } = useDatasetStore();

    const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'analyzing' | 'done' | 'error'>('idle');
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        if (acceptedFiles.length === 0) return;

        const file = acceptedFiles[0];
        const ext = file.name.split('.').pop()?.toLowerCase();

        if (!ext || !['csv', 'parquet', 'pq'].includes(ext)) {
            setErrorMessage('Unsupported file format. Please upload CSV or Parquet files.');
            setUploadStatus('error');
            return;
        }

        try {
            setUploadStatus('uploading');
            setUploading(true);
            setError(null);
            setErrorMessage(null);

            // Upload the file
            const datasetInfo = await apiClient.uploadDataset(file);

            setUploadStatus('analyzing');
            setAnalyzing(true);

            // Analyze the file
            const analysis = await apiClient.analyzeDataset(file, datasetInfo.dataset_id);

            // Update store
            addDataset({
                ...datasetInfo,
                status: 'analyzed',
            });
            setProfile(analysis.dataset_profile);
            setColumnAnalysis(analysis.column_analysis);

            setUploadStatus('done');

            // Reset status after a delay
            setTimeout(() => setUploadStatus('idle'), 3000);

        } catch (err) {
            const message = err instanceof Error ? err.message : 'Upload failed';
            setErrorMessage(message);
            setError(message);
            setUploadStatus('error');
        } finally {
            setUploading(false);
            setAnalyzing(false);
        }
    }, [addDataset, setProfile, setColumnAnalysis, setUploading, setAnalyzing, setError]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/csv': ['.csv'],
            'application/vnd.apache.parquet': ['.parquet', '.pq'],
        },
        maxFiles: 1,
        disabled: isUploading,
    });

    const getStatusIcon = () => {
        switch (uploadStatus) {
            case 'uploading':
            case 'analyzing':
                return <Loader2 className="upload-zone-icon animate-spin" size={48} />;
            case 'done':
                return <CheckCircle className="upload-zone-icon" size={48} style={{ color: 'var(--success)' }} />;
            case 'error':
                return <AlertCircle className="upload-zone-icon" size={48} style={{ color: 'var(--error)' }} />;
            default:
                return isDragActive
                    ? <FileSpreadsheet className="upload-zone-icon" size={48} style={{ color: 'var(--primary-400)' }} />
                    : <Upload className="upload-zone-icon" size={48} />;
        }
    };

    const getStatusText = () => {
        switch (uploadStatus) {
            case 'uploading':
                return 'Uploading file...';
            case 'analyzing':
                return 'Running stress analysis...';
            case 'done':
                return 'Analysis complete!';
            case 'error':
                return errorMessage || 'Upload failed';
            default:
                return isDragActive
                    ? 'Drop the file here'
                    : 'Drop CSV or Parquet file here, or click to browse';
        }
    };

    return (
        <div
            {...getRootProps()}
            className={`upload-zone ${isDragActive ? 'drag-over' : ''} ${uploadStatus === 'error' ? 'error' : ''}`}
            style={{
                borderColor: uploadStatus === 'error' ? 'var(--error)' : undefined,
                cursor: isUploading ? 'wait' : 'pointer',
            }}
        >
            <input {...getInputProps()} />

            {getStatusIcon()}

            <div className="upload-zone-text">{getStatusText()}</div>

            {uploadStatus === 'idle' && (
                <div className="upload-zone-hint">
                    Supported: .csv, .parquet, .pq | Max size: 100MB
                </div>
            )}
        </div>
    );
}
