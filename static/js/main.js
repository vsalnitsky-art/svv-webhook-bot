// Sleeper OB Bot - Main JavaScript

// Update time every second
function updateTime() {
    const timeEl = document.getElementById('current-time');
    if (timeEl) {
        const now = new Date();
        timeEl.textContent = now.toISOString().replace('T', ' ').substr(0, 19) + ' UTC';
    }
}

setInterval(updateTime, 1000);

// API Helper
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    const response = await fetch(endpoint, options);
    return response.json();
}

// Format numbers
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return Number(num).toFixed(decimals);
}

function formatMoney(num) {
    if (num === null || num === undefined) return '-';
    return '$' + Number(num).toFixed(2);
}

function formatPercent(num) {
    if (num === null || num === undefined) return '-';
    return Number(num).toFixed(2) + '%';
}

// Time ago
function timeAgo(date) {
    if (!date) return '-';
    
    const now = new Date();
    const past = new Date(date);
    const diffMs = now - past;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return diffMins + 'm ago';
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return diffHours + 'h ago';
    
    const diffDays = Math.floor(diffHours / 24);
    return diffDays + 'd ago';
}

// Notification helper
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Confirm dialog
function confirmAction(message) {
    return window.confirm(message);
}

// Auto-refresh stats (optional)
let autoRefreshInterval = null;

function startAutoRefresh(seconds = 30) {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        // Only refresh if on dashboard
        if (window.location.pathname === '/') {
            refreshDashboardStats();
        }
    }, seconds * 1000);
}

async function refreshDashboardStats() {
    try {
        const data = await apiCall('/api/stats');
        if (data.success) {
            // Update stats if elements exist
            const balanceEl = document.getElementById('balance');
            if (balanceEl && data.data.paper_balance) {
                balanceEl.textContent = formatMoney(data.data.paper_balance);
            }
        }
    } catch (err) {
        console.error('Failed to refresh stats:', err);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Start auto-refresh on dashboard
    if (window.location.pathname === '/') {
        startAutoRefresh(30);
    }
    
    // Add notification styles
    const style = document.createElement('style');
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 6px;
            background: #161b22;
            border: 1px solid #30363d;
            color: #c9d1d9;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        .notification.success {
            border-color: #3fb950;
            background: rgba(63, 185, 80, 0.1);
        }
        
        .notification.error {
            border-color: #f85149;
            background: rgba(248, 81, 73, 0.1);
        }
        
        .notification.warning {
            border-color: #d29922;
            background: rgba(210, 153, 34, 0.1);
        }
        
        .notification.fade-out {
            animation: fadeOut 0.3s ease forwards;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeOut {
            to {
                opacity: 0;
                transform: translateY(-20px);
            }
        }
    `;
    document.head.appendChild(style);
});

// WebSocket connection (for real-time updates)
let ws = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        ws.onclose = () => {
            console.log('WebSocket disconnected');
            // Reconnect after 5 seconds
            setTimeout(connectWebSocket, 5000);
        };
        
        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    } catch (err) {
        console.error('Failed to connect WebSocket:', err);
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'signal':
            showNotification(`New signal: ${data.symbol} ${data.direction}`, 'info');
            break;
        case 'trade_opened':
            showNotification(`Trade opened: ${data.symbol}`, 'success');
            break;
        case 'trade_closed':
            const pnlType = data.pnl >= 0 ? 'success' : 'warning';
            showNotification(`Trade closed: ${data.symbol} P&L: ${formatMoney(data.pnl)}`, pnlType);
            break;
        case 'sleeper_ready':
            showNotification(`Sleeper ready: ${data.symbol} 🔥`, 'info');
            break;
    }
}

// Export for use in templates
window.apiCall = apiCall;
window.formatNumber = formatNumber;
window.formatMoney = formatMoney;
window.formatPercent = formatPercent;
window.timeAgo = timeAgo;
window.showNotification = showNotification;
window.confirmAction = confirmAction;
