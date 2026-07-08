// lion_server/public/app.js

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dom = {
        wsStatusDot: document.getElementById('ws-status-dot'),
        wsStatusText: document.getElementById('ws-status-text'),
        valGeneration: document.getElementById('val-generation'),
        valTick: document.getElementById('val-tick'),
        valNeurons: document.getElementById('val-neurons'),
        valSynapses: document.getElementById('val-synapses'),
        valGestalt: document.getElementById('val-gestalt'),
        barGestalt: document.getElementById('bar-gestalt'),
        valStress: document.getElementById('val-stress'),
        barStress: document.getElementById('bar-stress'),
        valAction: document.getElementById('val-action'),
        immuneStatus: document.getElementById('immune-status'),
        valImmuneFixes: document.getElementById('val-immune-fixes'),
        logContainer: document.getElementById('log-container'),
    };

    let ws = null;
    let reconnectTimer = null;

    function connectWs() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            dom.wsStatusDot.className = 'dot pulse connected';
            dom.wsStatusText.textContent = 'Connected';
            clearTimeout(reconnectTimer);
        };

        ws.onclose = () => {
            dom.wsStatusDot.className = 'dot disconnected';
            dom.wsStatusText.textContent = 'Disconnected - Retrying...';
            // Auto reconnect
            reconnectTimer = setTimeout(connectWs, 2000);
        };

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                handleEvent(payload);
            } catch (err) {
                console.error("Failed to parse WS message", err);
            }
        };
    }

    function handleEvent(payload) {
        const type = payload.type;
        const data = payload.data;

        appendLog(type, data);

        switch (type) {
            case 'Welcome':
                dom.valGeneration.textContent = data.generation;
                dom.valTick.textContent = data.tick;
                dom.valNeurons.textContent = data.neurons;
                dom.valSynapses.textContent = data.synapses;
                break;

            case 'Tick':
                dom.valTick.textContent = data.tick;
                
                // Gestalt
                const gestaltStr = data.gestalt_norm.toFixed(2);
                dom.valGestalt.textContent = gestaltStr;
                // Max expected norm is roughly sqrt(FEATURE_SIZE), e.g. sqrt(32) ≈ 5.6
                // Map to 0-100%
                const gPercent = Math.min(100, (data.gestalt_norm / 5.6) * 100);
                dom.barGestalt.style.width = `${gPercent}%`;

                // Stress
                const stressStr = data.stress.toFixed(2);
                dom.valStress.textContent = stressStr;
                dom.barStress.style.width = `${data.stress * 100}%`;

                // Action
                if (dom.valAction.textContent !== data.action) {
                    dom.valAction.textContent = data.action;
                    // Flash effect
                    dom.valAction.classList.remove('flash');
                    void dom.valAction.offsetWidth; // trigger reflow
                    dom.valAction.classList.add('flash');
                }

                // Immune
                dom.valImmuneFixes.textContent = data.immune_fixes;
                break;

            case 'SleepCycle':
                dom.valGeneration.textContent = data.generation;
                break;

            case 'ImmuneAlert':
                // Temporarily flash immune card
                dom.immuneStatus.classList.add('alert');
                dom.immuneStatus.innerHTML = `
                    <div class="shield-icon">⚠️</div>
                    <h3>Intervention</h3>
                    <p>${data.total_fixes} corrections made.</p>
                `;
                setTimeout(() => {
                    dom.immuneStatus.classList.remove('alert');
                    dom.immuneStatus.innerHTML = `
                        <div class="shield-icon">🛡️</div>
                        <h3>Healthy</h3>
                        <p>No NaN/Inf interventions required.</p>
                    `;
                }, 3000);
                break;
                
            case 'Saved':
            case 'Loaded':
                if (data.neuron_count !== undefined) dom.valNeurons.textContent = data.neuron_count;
                if (data.synapse_count !== undefined) dom.valSynapses.textContent = data.synapse_count;
                break;
        }
    }

    function appendLog(type, data) {
        const div = document.createElement('div');
        div.className = `log-entry ${type}`;
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        let detailStr = '';
        if (type === 'Tick') detailStr = `Action: ${data.action} | Gestalt: ${data.gestalt_norm.toFixed(2)}`;
        else if (type === 'SleepCycle') detailStr = `Gen: ${data.generation} | Evol: ${data.evolution_occurred} | Fit: ${data.sovereign_fitness.toFixed(2)}`;
        else if (type === 'Saved' || type === 'Loaded') detailStr = `Path: ${data.path}`;
        else detailStr = JSON.stringify(data);

        div.innerHTML = `
            <span class="log-time">${timeStr}</span>
            <span class="log-type">[${type}]</span>
            <span class="log-detail">${detailStr}</span>
        `;

        dom.logContainer.appendChild(div);
        
        // Auto-scroll
        dom.logContainer.scrollTop = dom.logContainer.scrollHeight;

        // Keep log size manageable
        while (dom.logContainer.children.length > 50) {
            dom.logContainer.removeChild(dom.logContainer.firstChild);
        }
    }

    // Start
    connectWs();
});
