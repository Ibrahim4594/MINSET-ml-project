/**
 * Frontend State Management
 * Centralized state management for the drawing application
 */

class AppState {
    constructor() {
        this.state = {
            // Drawing state
            drawing: {
                mode: 'pen', // 'pen' or 'eraser'
                brushSize: 12,
                history: [],
                historyStep: -1,
                isDrawing: false,
                lastX: 0,
                lastY: 0,
            },
            // Prediction state
            prediction: {
                digit: null,
                confidence: null,
                top3: [],
                inferenceTime: null,
                fromCache: false,
                timestamp: null,
            },
            // UI state
            ui: {
                isLoading: false,
                realtimeEnabled: false,
                status: '',
                statusType: 'info', // 'info', 'success', 'error'
            },
            // Settings
            settings: {
                autoPredict: false,
                showGridlines: false,
                theme: 'light',
            }
        };

        // Track history of predictions
        this.predictionHistory = [];
        this.maxHistorySize = 50;

        // Listeners
        this.listeners = new Map();
    }

    // State getters
    getState(path) {
        const parts = path.split('.');
        let current = this.state;
        for (const part of parts) {
            current = current[part];
            if (current === undefined) return undefined;
        }
        return current;
    }

    // State setters
    setState(path, value) {
        const parts = path.split('.');
        const key = parts.pop();
        let current = this.state;

        for (const part of parts) {
            if (!current[part]) {
                current[part] = {};
            }
            current = current[part];
        }

        current[key] = value;
        this.notifyListeners(path);
    }

    // Update nested state
    updateState(path, updates) {
        const current = this.getState(path);
        const updated = { ...current, ...updates };
        this.setState(path, updated);
    }

    // Subscribe to state changes
    subscribe(path, callback) {
        if (!this.listeners.has(path)) {
            this.listeners.set(path, []);
        }
        this.listeners.get(path).push(callback);

        // Return unsubscribe function
        return () => {
            const callbacks = this.listeners.get(path);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        };
    }

    // Notify listeners
    notifyListeners(path) {
        if (this.listeners.has(path)) {
            const value = this.getState(path);
            this.listeners.get(path).forEach(callback => callback(value));
        }
    }

    // Drawing methods
    updateDrawingMode(mode) {
        this.setState('drawing.mode', mode);
    }

    updateBrushSize(size) {
        this.setState('drawing.brushSize', size);
    }

    setDrawing(isDrawing) {
        this.setState('drawing.isDrawing', isDrawing);
    }

    saveDrawingState(canvasDataUrl) {
        const history = this.getState('drawing.history');
        let step = this.getState('drawing.historyStep');

        step++;
        if (step < history.length) {
            history.length = step;
        }
        history.push(canvasDataUrl);

        this.updateState('drawing', {
            history: history,
            historyStep: step
        });
    }

    canUndo() {
        const historyStep = this.getState('drawing.historyStep');
        return historyStep > 0;
    }

    canRedo() {
        const history = this.getState('drawing.history');
        const historyStep = this.getState('drawing.historyStep');
        return historyStep < history.length - 1;
    }

    getUndoState() {
        if (!this.canUndo()) return null;
        const history = this.getState('drawing.history');
        const historyStep = this.getState('drawing.historyStep') - 1;
        return history[historyStep];
    }

    getRedoState() {
        if (!this.canRedo()) return null;
        const history = this.getState('drawing.history');
        const historyStep = this.getState('drawing.historyStep') + 1;
        return history[historyStep];
    }

    undo() {
        if (this.canUndo()) {
            const step = this.getState('drawing.historyStep') - 1;
            this.setState('drawing.historyStep', step);
            return this.getState('drawing.history')[step];
        }
        return null;
    }

    redo() {
        if (this.canRedo()) {
            const step = this.getState('drawing.historyStep') + 1;
            this.setState('drawing.historyStep', step);
            return this.getState('drawing.history')[step];
        }
        return null;
    }

    clearDrawing() {
        this.setState('drawing', {
            mode: 'pen',
            brushSize: 12,
            history: [],
            historyStep: -1,
            isDrawing: false,
        });
    }

    // Prediction methods
    setPrediction(digit, confidence, top3, inferenceTime, fromCache = false) {
        const prediction = {
            digit,
            confidence,
            top3,
            inferenceTime,
            fromCache,
            timestamp: new Date().toISOString(),
        };

        this.setState('prediction', prediction);

        // Add to history
        this.predictionHistory.push(prediction);
        if (this.predictionHistory.length > this.maxHistorySize) {
            this.predictionHistory.shift();
        }
    }

    getPredictionHistory() {
        return [...this.predictionHistory];
    }

    clearPrediction() {
        this.setState('prediction', {
            digit: null,
            confidence: null,
            top3: [],
            inferenceTime: null,
            fromCache: false,
            timestamp: null,
        });
    }

    // UI methods
    setLoading(isLoading) {
        this.setState('ui.isLoading', isLoading);
    }

    setRealtimeEnabled(enabled) {
        this.setState('ui.realtimeEnabled', enabled);
    }

    setStatus(message, type = 'info') {
        this.updateState('ui', {
            status: message,
            statusType: type
        });
    }

    // Settings methods
    updateSetting(key, value) {
        this.setState(`settings.${key}`, value);
    }

    // Reset state
    reset() {
        this.state = {
            drawing: {
                mode: 'pen',
                brushSize: 12,
                history: [],
                historyStep: -1,
                isDrawing: false,
                lastX: 0,
                lastY: 0,
            },
            prediction: {
                digit: null,
                confidence: null,
                top3: [],
                inferenceTime: null,
                fromCache: false,
                timestamp: null,
            },
            ui: {
                isLoading: false,
                realtimeEnabled: false,
                status: '',
                statusType: 'info',
            },
            settings: {
                autoPredict: false,
                showGridlines: false,
                theme: 'light',
            }
        };
        this.predictionHistory = [];
    }

    // Export state for debugging
    export() {
        return {
            state: JSON.parse(JSON.stringify(this.state)),
            predictionHistory: this.predictionHistory,
        };
    }
}

// Create global app state instance
const appState = new AppState();
