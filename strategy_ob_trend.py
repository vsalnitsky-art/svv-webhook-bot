{% extends "base.html" %}

{% block title %}Whale Strategy{% endblock %}

{% block head %}
<style>
    /* ===== СТИЛІ ДЛЯ СОРТУВАННЯ ===== */
    th.sortable { 
        cursor: pointer; 
        user-select: none; 
        position: relative;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    
    th.sortable:hover { 
        background-color: #e8f0f7;
        color: var(--accent, #3b82f6);
    }

    th.sortable::after { 
        content: ' ↕'; 
        font-size: 0.75em; 
        opacity: 0.3;
        margin-left: 4px;
    }
    
    th.sortable:hover::after {
        opacity: 0.6;
    }
    
    th.asc::after { 
        content: ' ↑'; 
        opacity: 1; 
        color: var(--accent, #3b82f6);
        font-weight: bold;
    }
    
    th.desc::after { 
        content: ' ↓'; 
        opacity: 1; 
        color: var(--accent, #3b82f6);
        font-weight: bold;
    }
    
    th.sortable.asc,
    th.sortable.desc {
        background-color: #eff6ff;
        color: var(--accent, #3b82f6);
    }

    #whaleTable tbody tr {
        transition: background-color 0.15s ease;
    }
    #whaleTable tbody tr:hover {
        background-color: #f8fafc;
    }
</style>
{% endblock %}

{% block content %}
<div class="container-fluid mt-4">
    <div class="d-flex justify-content-between align-items-center mb-4">
        <div>
            <h3 class="fw-bold m-0 text-primary">
                <i class="ri-water-flash-fill me-2"></i>Whale Strategy
            </h3>
            <p class="text-muted small m-0">Автономний модуль пошуку акумуляції (Whale)</p>
        </div>
        <div>
            <button onclick="startWhaleScan()" id="scanBtn" class="btn btn-primary fw-bold shadow-sm" {{ 'disabled' if is_scanning }}>
                {% if is_scanning %}
                <span class="spinner-border spinner-border-sm me-2"></span> Сканування {{ progress }}%
                {% else %}
                <i class="ri-radar-line me-2"></i> ЗАПУСТИТИ СКАНЕР
                {% endif %}
            </button>
        </div>
    </div>

    <div class="card border-0 shadow-sm mb-4">
        <div class="card-body bg-light rounded-3">
            <div class="row g-3 align-items-end">
                <div class="col-md-2">
                    <label class="small fw-bold text-muted text-uppercase">Таймфрейм</label>
                    <select class="form-select form-select-sm" id="whale_tf">
                        <option value="15">15 Хвилин</option>
                        <option value="60" selected>1 Година</option>
                        <option value="240">4 Години</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="small fw-bold text-muted text-uppercase">Глибина</label>
                    <select class="form-select form-select-sm" id="whale_limit">
                        <option value="20">Топ 20</option>
                        <option value="50" selected>Топ 50</option>
                        <option value="100">Топ 100</option>
                    </select>
                </div>
                
                <div class="col-md-2">
                    <label class="small fw-bold text-muted text-uppercase">RSI Фільтр</label>
                    <div class="form-check form-switch mt-1">
                        <input class="form-check-input" type="checkbox" id="rsi_filter_switch" {{ 'checked' if conf.get('whale_rsi_filter_enabled') }}>
                        <label class="form-check-label small" for="rsi_filter_switch">Увімкнено</label>
                    </div>
                </div>
                <div class="col-md-1">
                    <label class="small fw-bold text-muted text-uppercase">RSI ≤</label>
                    <input type="number" class="form-control form-control-sm" id="rsi_min" value="{{ conf.get('whale_rsi_min', 30) }}" min="10" max="50">
                </div>
                <div class="col-md-1">
                    <label class="small fw-bold text-muted text-uppercase">RSI ≥</label>
                    <input type="number" class="form-control form-control-sm" id="rsi_max" value="{{ conf.get('whale_rsi_max', 70) }}" min="50" max="90">
                </div>
                
                <div class="col-md-2">
                    <div class="small text-muted text-uppercase fw-bold">Статус</div>
                    <div class="fw-bold text-dark" id="status-text">{{ status }}</div>
                </div>
                <div class="col-md-2">
                    <div class="small text-muted text-uppercase fw-bold">Останній скан</div>
                    <div class="fw-bold text-primary">{{ last_time or "-" }}</div>
                </div>
            </div>
            
            <div class="row mt-2" id="rsi_hint" style="display: none;">
                <div class="col-12">
                    <div class="alert alert-info py-2 px-3 small mb-0">
                        <i class="ri-information-line me-1"></i>
                        <strong>RSI Фільтр:</strong> Шукає монети з RSI ≤ <span id="hint_min">30</span> (перепроданість) або RSI ≥ <span id="hint_max">70</span> (перекупленість). 
                        <br><small class="text-muted">Пороги автоматично підлаштовуються під таймфрейм: 15хв (25/75), 1год (30/70), 4год (35/65)</small>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="card border-0 shadow-sm">
        <div class="card-header bg-white py-3 border-bottom d-flex justify-content-between align-items-center">
            <h6 class="fw-bold m-0"><i class="ri-database-2-line me-2"></i>Результати</h6>
            <span class="badge bg-light text-dark border">{{ history|length }} записів</span>
        </div>
        <div class="table-responsive">
            <table class="table table-hover align-middle mb-0" id="whaleTable">
                <thead class="bg-light text-uppercase small text-muted" style="font-size: 11px; letter-spacing: 1px;">
                    <tr>
                        <th class="ps-4 py-3 border-bottom sortable" data-column="0">Час</th>
                        <th class="py-3 border-bottom sortable" data-column="1">Актив</th>
                        <th class="py-3 border-bottom sortable" data-column="2">Ціна</th>
                        <th class="py-3 border-bottom sortable" data-column="3">RSI</th>
                        <th class="py-3 border-bottom sortable" data-column="4">Score</th>
                        <th class="py-3 border-bottom sortable" data-column="5">Стиснення</th>
                        <th class="py-3 border-bottom">Причина</th>
                        <th class="py-3 pe-4 text-end border-bottom">Дія</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in history %}
                    <tr>
                        <td class="ps-4 text-muted small font-mono" data-value="{{ row.time }}">{{ row.time }}</td>
                        <td class="fw-bold text-primary" data-value="{{ row.symbol }}">{{ row.symbol }}</td>
                        <td class="font-mono" data-value="{{ row.price }}">{{ row.price }}</td>
                        <td data-value="{{ row.rsi or 0 }}">
                            {% if row.rsi %}
                                <span class="badge {{ 'bg-danger' if row.rsi >= 70 else 'bg-success' if row.rsi <= 30 else 'bg-secondary' }}">
                                    {{ row.rsi }}
                                </span>
                            {% else %}
                                <span class="text-muted">-</span>
                            {% endif %}
                        </td>
                        <td data-value="{{ row.score }}">
                            {% if row.score >= 80 %}
                                <span class="badge bg-success">STRONG ({{ row.score }})</span>
                            {% elif row.score >= 70 %}
                                <span class="badge bg-info text-dark">GOOD ({{ row.score }})</span>
                            {% else %}
                                <span class="badge bg-warning text-dark">WEAK ({{ row.score }})</span>
                            {% endif %}
                        </td>
                        <td class="font-mono" data-value="{{ row.squeeze }}">{{ row.squeeze }}</td>
                        <td class="small text-muted">{{ row.details }}</td>
                        <td class="text-end pe-4">
                            <a href="https://www.bybit.com/trade/usdt/{{ row.symbol }}" target="_blank" class="btn btn-sm btn-outline-light text-dark border">
                                <i class="ri-line-chart-line"></i>
                            </a>
                        </td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="8" class="text-center py-5 text-muted">
                            <i class="ri-search-eye-line fs-1 opacity-25"></i>
                            <p class="mt-2">Даних немає. Налаштуйте параметри та натисніть "Запустити Сканер".</p>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
// ===== RSI ПОРОГИ ДЛЯ РІЗНИХ ТАЙМФРЕЙМІВ =====
// Рекомендовані значення для кожного ТФ (синхронізовані з TradingView)
var RSI_THRESHOLDS = {
    '15':  { min: 25, max: 75 },  // 15хв - волатильний, екстремальні пороги
    '60':  { min: 30, max: 70 },  // 1год - стандартні пороги
    '240': { min: 35, max: 65 }   // 4год - консервативні пороги
};

// При зміні таймфрейму - автоматично оновлюємо RSI пороги
document.getElementById('whale_tf').addEventListener('change', function() {
    var tf = this.value;
    var thresholds = RSI_THRESHOLDS[tf] || RSI_THRESHOLDS['60'];
    
    document.getElementById('rsi_min').value = thresholds.min;
    document.getElementById('rsi_max').value = thresholds.max;
    
    // Оновлюємо підказку
    updateRsiHint(thresholds.min, thresholds.max);
});

function updateRsiHint(min, max) {
    var hintMin = document.getElementById('hint_min');
    var hintMax = document.getElementById('hint_max');
    if (hintMin) hintMin.textContent = min;
    if (hintMax) hintMax.textContent = max;
}

// ===== RSI ФІЛЬТР TOGGLE =====
document.getElementById('rsi_filter_switch').addEventListener('change', function() {
    document.getElementById('rsi_hint').style.display = this.checked ? 'block' : 'none';
});
if (document.getElementById('rsi_filter_switch').checked) {
    document.getElementById('rsi_hint').style.display = 'block';
}

// Ініціалізація RSI порогів при завантаженні сторінки
document.addEventListener('DOMContentLoaded', function() {
    var tf = document.getElementById('whale_tf').value;
    var thresholds = RSI_THRESHOLDS[tf] || RSI_THRESHOLDS['60'];
    document.getElementById('rsi_min').value = thresholds.min;
    document.getElementById('rsi_max').value = thresholds.max;
    updateRsiHint(thresholds.min, thresholds.max);
});

// ===== СОРТУВАННЯ ТАБЛИЦІ =====
class TableSorter {
    constructor(tableId, defaultColumn = null, defaultDirection = 'desc') {
        this.table = document.getElementById(tableId);
        if (!this.table) return;
        
        this.sortColumn = null;
        this.sortDirection = 'asc';
        this.storageKey = 'table_sort_' + tableId;
        this.defaultColumn = defaultColumn;
        this.defaultDirection = defaultDirection;
        
        this.attachSortHandlers();
    }
    
    attachSortHandlers() {
        var self = this;
        document.querySelectorAll('#' + this.table.id + ' th.sortable').forEach(function(th) {
            th.addEventListener('click', function() {
                var colIndex = parseInt(th.getAttribute('data-column'));
                self.sort(colIndex, th, true);
            });
        });
    }
    
    sort(columnIndex, headerElement, toggle) {
        var self = this;
        var rows = Array.from(this.table.querySelectorAll('tbody tr'));
        
        if (rows.length === 0 || (rows.length === 1 && rows[0].querySelector('td[colspan]'))) {
            return;
        }
        
        if (toggle && this.sortColumn === columnIndex) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else if (toggle) {
            this.sortDirection = 'desc';
        }
        
        document.querySelectorAll('#' + this.table.id + ' th.sortable').forEach(function(th) {
            th.classList.remove('asc', 'desc');
        });
        
        this.sortColumn = columnIndex;
        
        rows.sort(function(rowA, rowB) {
            var cellA = rowA.children[columnIndex];
            var cellB = rowB.children[columnIndex];
            
            if (!cellA || !cellB) return 0;
            
            var valA = cellA.getAttribute('data-value') || cellA.innerText.trim();
            var valB = cellB.getAttribute('data-value') || cellB.innerText.trim();
            
            var numA = parseFloat(valA);
            var numB = parseFloat(valB);
            
            if (!isNaN(numA) && !isNaN(numB)) {
                return self.sortDirection === 'asc' ? numA - numB : numB - numA;
            } else {
                valA = valA.toLowerCase();
                valB = valB.toLowerCase();
                if (valA === valB) return 0;
                if (self.sortDirection === 'asc') {
                    return valA > valB ? 1 : -1;
                } else {
                    return valA < valB ? 1 : -1;
                }
            }
        });
        
        var tbody = this.table.querySelector('tbody');
        rows.forEach(function(row) { tbody.appendChild(row); });
        
        headerElement.classList.add(this.sortDirection);
        this.saveSortState();
    }
    
    saveSortState() {
        var state = { column: this.sortColumn, direction: this.sortDirection };
        localStorage.setItem(this.storageKey, JSON.stringify(state));
    }
    
    loadSortState() {
        var saved = localStorage.getItem(this.storageKey);
        if (saved) {
            try { return JSON.parse(saved); } catch (e) { return null; }
        }
        return null;
    }
    
    applyInitialSort() {
        var savedState = this.loadSortState();
        var columnToSort, directionToUse;
        
        if (savedState && savedState.column !== null) {
            columnToSort = savedState.column;
            directionToUse = savedState.direction;
        } else if (this.defaultColumn !== null) {
            columnToSort = this.defaultColumn;
            directionToUse = this.defaultDirection;
        } else {
            return;
        }
        
        var headers = this.table.querySelectorAll('th.sortable');
        var headerToSort = null;
        headers.forEach(function(th) {
            if (parseInt(th.getAttribute('data-column')) === columnToSort) {
                headerToSort = th;
            }
        });
        
        if (headerToSort) {
            this.sortDirection = directionToUse;
            this.sort(columnToSort, headerToSort, false);
        }
    }
}

// Ініціалізація сортування (по Score DESC за замовчуванням)
document.addEventListener('DOMContentLoaded', function() {
    var sorter = new TableSorter('whaleTable', 4, 'desc');
    sorter.applyInitialSort();
});

// ===== ЗАПУСК СКАНЕРА =====
function startWhaleScan() {
    var btn = document.getElementById('scanBtn');
    var tf = document.getElementById('whale_tf').value;
    var limit = document.getElementById('whale_limit').value;
    
    var rsiEnabled = document.getElementById('rsi_filter_switch').checked;
    var rsiMin = document.getElementById('rsi_min').value;
    var rsiMax = document.getElementById('rsi_max').value;
    
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Запуск...';
    
    fetch('/whale/scan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            timeframe: tf, 
            limit: limit,
            rsi_filter_enabled: rsiEnabled,
            rsi_min: rsiMin,
            rsi_max: rsiMax
        })
    })
    .then(function(r) { return r.json(); })
    .then(function(d) {
        if(d.status === 'started') {
            setTimeout(function() { location.reload(); }, 500);
        } else {
            alert('Сканер вже працює!');
            location.reload();
        }
    })
    .catch(function(err) {
        alert('Помилка з\'єднання');
        btn.disabled = false;
        btn.innerHTML = '<i class="ri-radar-line me-2"></i> ЗАПУСТИТИ СКАНЕР';
    });
}

// Автооновлення при скануванні
{% if is_scanning %}
setTimeout(function() { location.reload(); }, 2000);
{% endif %}
</script>
{% endblock %}
