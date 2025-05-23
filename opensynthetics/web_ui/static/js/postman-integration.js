/**
 * OpenSynthetics Postman Integration
 * API collection generation, testing, and API key management
 */

class PostmanIntegration {
    constructor(enhancedFeatures) {
        this.enhancedFeatures = enhancedFeatures;
        this.apiKeys = new Map();
        this.collections = new Map();
        this.environments = new Map();
        
        this.init();
    }

    async init() {
        this.createPostmanPage();
        this.createAPIKeysPage();
        this.loadAPIKeys();
        this.loadCollections();
        this.setupEventListeners();
    }

    createPostmanPage() {
        const pageContent = `
            <div id="postman-page" class="page">
                <div class="page-header">
                    <h1 class="page-title">Postman Integration</h1>
                    <p class="page-subtitle">Generate API collections, test endpoints, and manage integrations</p>
                </div>

                <!-- API Collection Generator -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">API Collection Generator</h3>
                        <button class="btn btn-primary" onclick="postmanIntegration.generateCollection()">
                            <i class="fas fa-file-export"></i>
                            Generate Collection
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-2">
                            <div>
                                <h4>OpenSynthetics API</h4>
                                <p>Generate a complete Postman collection for the OpenSynthetics API</p>
                                <div style="margin-top: var(--spacing-md);">
                                    <label class="form-label">Base URL</label>
                                    <input type="url" class="form-input" id="api-base-url" value="${window.location.origin}" readonly>
                                </div>
                                <div style="margin-top: var(--spacing-md);">
                                    <label class="form-label">Include Endpoints</label>
                                    <div style="margin-top: var(--spacing-sm);">
                                        <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                            <input type="checkbox" checked id="include-workspaces"> Workspaces
                                        </label>
                                        <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                            <input type="checkbox" checked id="include-datasets"> Datasets
                                        </label>
                                        <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                            <input type="checkbox" checked id="include-generation"> Data Generation
                                        </label>
                                        <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                            <input type="checkbox" checked id="include-training"> Training
                                        </label>
                                        <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                            <input type="checkbox" checked id="include-validation"> Validation
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <h4>Collection Settings</h4>
                                <div class="form-group">
                                    <label class="form-label">Collection Name</label>
                                    <input type="text" class="form-input" id="collection-name" value="OpenSynthetics API" placeholder="Collection name">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Description</label>
                                    <textarea class="form-textarea" id="collection-description" rows="3" placeholder="Collection description">Complete API collection for OpenSynthetics platform</textarea>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Version</label>
                                    <input type="text" class="form-input" id="collection-version" value="1.0.0" placeholder="1.0.0">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Existing Collections -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Saved Collections</h3>
                        <button class="btn btn-secondary" onclick="postmanIntegration.refreshCollections()">
                            <i class="fas fa-sync"></i>
                            Refresh
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="collections-list">
                            <div class="text-center">
                                <i class="fas fa-spinner fa-spin fa-2x"></i>
                                <p style="margin-top: var(--spacing-md);">Loading collections...</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- API Testing -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">API Testing</h3>
                        <button class="btn btn-primary" onclick="postmanIntegration.openTestRunner()">
                            <i class="fas fa-play"></i>
                            Run Tests
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-2">
                            <div>
                                <h4>Quick Test</h4>
                                <p>Test a single API endpoint</p>
                                <div class="form-group">
                                    <label class="form-label">Method</label>
                                    <select class="form-select" id="test-method">
                                        <option value="GET">GET</option>
                                        <option value="POST">POST</option>
                                        <option value="PUT">PUT</option>
                                        <option value="DELETE">DELETE</option>
                                        <option value="PATCH">PATCH</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Endpoint</label>
                                    <input type="text" class="form-input" id="test-endpoint" placeholder="/api/v1/workspaces">
                                </div>
                                <button class="btn btn-secondary" onclick="postmanIntegration.runQuickTest()">
                                    <i class="fas fa-paper-plane"></i>
                                    Send Request
                                </button>
                            </div>
                            <div>
                                <h4>Test Results</h4>
                                <div id="test-results" style="background: var(--surface-secondary); padding: var(--spacing-md); border-radius: var(--border-radius); min-height: 150px; font-family: var(--font-mono); font-size: 0.9rem; white-space: pre-wrap;">
                                    No tests run yet
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Postman Workspace Integration -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Postman Workspace</h3>
                        <div id="postman-auth-status">
                            <span class="status-indicator" id="postman-status-indicator"></span>
                            <span id="postman-status-text">Not connected</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="postman-signed-out">
                            <p>Connect to your Postman workspace to automatically sync collections and environments.</p>
                            <div style="margin-top: var(--spacing-lg);">
                                <button class="btn btn-primary" onclick="postmanIntegration.connectPostman()">
                                    <i class="fas fa-paper-plane"></i>
                                    Connect Postman
                                </button>
                                <button class="btn btn-secondary" onclick="postmanIntegration.showPostmanConfig()">
                                    <i class="fas fa-cog"></i>
                                    Configure API Key
                                </button>
                            </div>
                        </div>
                        
                        <div id="postman-signed-in" class="hidden">
                            <div class="grid grid-2">
                                <div>
                                    <h4>Workspace Info</h4>
                                    <div id="workspace-info">
                                        <div style="margin-bottom: var(--spacing-sm);">
                                            <strong>Name:</strong> <span id="workspace-name">-</span>
                                        </div>
                                        <div style="margin-bottom: var(--spacing-sm);">
                                            <strong>Type:</strong> <span id="workspace-type">-</span>
                                        </div>
                                        <div style="margin-bottom: var(--spacing-sm);">
                                            <strong>Collections:</strong> <span id="workspace-collections">-</span>
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <h4>Actions</h4>
                                    <div style="display: flex; flex-direction: column; gap: var(--spacing-sm);">
                                        <button class="btn btn-secondary" onclick="postmanIntegration.syncToPostman()">
                                            <i class="fas fa-sync"></i>
                                            Sync to Postman
                                        </button>
                                        <button class="btn btn-secondary" onclick="postmanIntegration.createEnvironment()">
                                            <i class="fas fa-plus"></i>
                                            Create Environment
                                        </button>
                                        <button class="btn btn-danger" onclick="postmanIntegration.disconnectPostman()">
                                            <i class="fas fa-unlink"></i>
                                            Disconnect
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const dynamicContent = document.getElementById('dynamic-content');
        if (dynamicContent) {
            dynamicContent.insertAdjacentHTML('beforeend', pageContent);
        }
    }

    createAPIKeysPage() {
        const pageContent = `
            <div id="api-keys-page" class="page">
                <div class="page-header">
                    <h1 class="page-title">API Key Management</h1>
                    <p class="page-subtitle">Manage API keys for external integrations and services</p>
                </div>

                <!-- API Key Creation -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Create New API Key</h3>
                        <button class="btn btn-primary" onclick="postmanIntegration.openCreateAPIKeyModal()">
                            <i class="fas fa-plus"></i>
                            Create API Key
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-3">
                            <div class="card" style="text-align: center; padding: var(--spacing-lg);">
                                <i class="fas fa-paper-plane fa-2x" style="color: var(--primary-color); margin-bottom: var(--spacing-md);"></i>
                                <h4>Postman API</h4>
                                <p style="color: var(--text-secondary);">Integrate with Postman workspaces</p>
                                <button class="btn btn-secondary" onclick="postmanIntegration.createPostmanAPIKey()">
                                    <i class="fas fa-plus"></i>
                                    Add Key
                                </button>
                            </div>
                            <div class="card" style="text-align: center; padding: var(--spacing-lg);">
                                <i class="fab fa-google fa-2x" style="color: var(--primary-color); margin-bottom: var(--spacing-md);"></i>
                                <h4>Google APIs</h4>
                                <p style="color: var(--text-secondary);">Google Drive, Sheets, etc.</p>
                                <button class="btn btn-secondary" onclick="postmanIntegration.createGoogleAPIKey()">
                                    <i class="fas fa-plus"></i>
                                    Add Key
                                </button>
                            </div>
                            <div class="card" style="text-align: center; padding: var(--spacing-lg);">
                                <i class="fas fa-cloud fa-2x" style="color: var(--primary-color); margin-bottom: var(--spacing-md);"></i>
                                <h4>Cloud Services</h4>
                                <p style="color: var(--text-secondary);">AWS, Azure, GCP</p>
                                <button class="btn btn-secondary" onclick="postmanIntegration.createCloudAPIKey()">
                                    <i class="fas fa-plus"></i>
                                    Add Key
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- API Keys List -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Existing API Keys</h3>
                        <button class="btn btn-secondary" onclick="postmanIntegration.refreshAPIKeys()">
                            <i class="fas fa-sync"></i>
                            Refresh
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="api-keys-list">
                            <div class="text-center">
                                <i class="fas fa-spinner fa-spin fa-2x"></i>
                                <p style="margin-top: var(--spacing-md);">Loading API keys...</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- API Usage Analytics -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Usage Analytics</h3>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-2">
                            <div>
                                <h4>Usage Statistics</h4>
                                <div id="usage-stats">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                                        <span>Total Requests Today:</span>
                                        <span id="requests-today">-</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                                        <span>Total Requests This Month:</span>
                                        <span id="requests-month">-</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                                        <span>Active API Keys:</span>
                                        <span id="active-keys">-</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                                        <span>Failed Requests (24h):</span>
                                        <span id="failed-requests">-</span>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <h4>Rate Limits</h4>
                                <div id="rate-limits">
                                    <div class="progress-container" style="margin-bottom: var(--spacing-md);">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-xs);">
                                            <span>Hourly Limit</span>
                                            <span id="hourly-usage">-</span>
                                        </div>
                                        <div class="progress-bar">
                                            <div class="progress-fill" id="hourly-progress" style="width: 0%;"></div>
                                        </div>
                                    </div>
                                    <div class="progress-container">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-xs);">
                                            <span>Daily Limit</span>
                                            <span id="daily-usage">-</span>
                                        </div>
                                        <div class="progress-bar">
                                            <div class="progress-fill" id="daily-progress" style="width: 0%;"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const dynamicContent = document.getElementById('dynamic-content');
        if (dynamicContent) {
            dynamicContent.insertAdjacentHTML('beforeend', pageContent);
        }
    }

    setupEventListeners() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                switch (e.key) {
                    case 'P':
                        e.preventDefault();
                        this.enhancedFeatures.app.showPage('postman');
                        break;
                    case 'K':
                        e.preventDefault();
                        this.enhancedFeatures.app.showPage('api-keys');
                        break;
                }
            }
        });
    }

    async loadAPIKeys() {
        try {
            const response = await fetch('/api/v1/api-keys');
            const data = await response.json();
            this.renderAPIKeys(data.keys || []);
            this.updateUsageStats(data.stats || {});
        } catch (error) {
            console.error('Failed to load API keys:', error);
            this.renderAPIKeys([]);
        }
    }

    renderAPIKeys(keys) {
        const container = document.getElementById('api-keys-list');
        
        if (keys.length === 0) {
            container.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-key fa-3x" style="color: var(--text-tertiary); margin-bottom: var(--spacing-lg);"></i>
                    <h3>No API keys configured</h3>
                    <p style="color: var(--text-secondary);">Create your first API key to get started with integrations</p>
                </div>
            `;
            return;
        }

        container.innerHTML = keys.map(key => this.createAPIKeyHTML(key)).join('');
    }

    createAPIKeyHTML(key) {
        const statusColor = key.status === 'active' ? 'var(--success-color)' : 
                           key.status === 'expired' ? 'var(--danger-color)' : 
                           'var(--warning-color)';

        return `
            <div class="card" style="margin-bottom: var(--spacing-md);">
                <div class="card-header">
                    <div style="display: flex; align-items: center; gap: var(--spacing-md);">
                        <i class="${this.getServiceIcon(key.service)}" style="font-size: 1.5rem; color: var(--primary-color);"></i>
                        <div>
                            <h4 style="margin: 0;">${key.name}</h4>
                            <div style="font-size: 0.9rem; color: var(--text-secondary);">${key.service}</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: var(--spacing-md);">
                        <span class="status-indicator" style="background-color: ${statusColor};"></span>
                        <span style="color: ${statusColor}; font-weight: 500;">${key.status}</span>
                        <button class="btn btn-sm btn-icon btn-secondary" onclick="postmanIntegration.toggleAPIKeyActions('${key.id}')">
                            <i class="fas fa-ellipsis-v"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body">
                    <div class="grid grid-2">
                        <div>
                            <div style="margin-bottom: var(--spacing-sm);">
                                <strong>Created:</strong> ${new Date(key.created_at).toLocaleDateString()}
                            </div>
                            <div style="margin-bottom: var(--spacing-sm);">
                                <strong>Last Used:</strong> ${key.last_used ? new Date(key.last_used).toLocaleDateString() : 'Never'}
                            </div>
                            <div style="margin-bottom: var(--spacing-sm);">
                                <strong>Requests (24h):</strong> ${key.usage_24h || 0}
                            </div>
                        </div>
                        <div>
                            <div style="margin-bottom: var(--spacing-sm);">
                                <strong>Expires:</strong> ${key.expires_at ? new Date(key.expires_at).toLocaleDateString() : 'Never'}
                            </div>
                            <div style="margin-bottom: var(--spacing-sm);">
                                <strong>Rate Limit:</strong> ${key.rate_limit || 'Unlimited'} req/hour
                            </div>
                            <div style="margin-bottom: var(--spacing-sm);">
                                <strong>Scope:</strong> ${key.scope || 'Full access'}
                            </div>
                        </div>
                    </div>
                    
                    <div class="api-key-value" style="margin-top: var(--spacing-lg); background: var(--surface-secondary); padding: var(--spacing-md); border-radius: var(--border-radius); font-family: var(--font-mono); font-size: 0.9rem; position: relative;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span id="key-value-${key.id}" style="user-select: all;">${key.masked_key}</span>
                            <div style="display: flex; gap: var(--spacing-sm);">
                                <button class="btn btn-sm btn-secondary" onclick="postmanIntegration.copyAPIKey('${key.id}')">
                                    <i class="fas fa-copy"></i>
                                </button>
                                <button class="btn btn-sm btn-secondary" onclick="postmanIntegration.revealAPIKey('${key.id}')">
                                    <i class="fas fa-eye"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="api-key-actions hidden" id="actions-${key.id}" style="margin-top: var(--spacing-lg); display: flex; gap: var(--spacing-sm);">
                        <button class="btn btn-sm btn-secondary" onclick="postmanIntegration.editAPIKey('${key.id}')">
                            <i class="fas fa-edit"></i>
                            Edit
                        </button>
                        <button class="btn btn-sm btn-warning" onclick="postmanIntegration.regenerateAPIKey('${key.id}')">
                            <i class="fas fa-sync"></i>
                            Regenerate
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="postmanIntegration.revokeAPIKey('${key.id}')">
                            <i class="fas fa-trash"></i>
                            Revoke
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    getServiceIcon(service) {
        const iconMap = {
            'postman': 'fas fa-paper-plane',
            'google': 'fab fa-google',
            'aws': 'fab fa-aws',
            'azure': 'fab fa-microsoft',
            'github': 'fab fa-github',
            'slack': 'fab fa-slack',
            'discord': 'fab fa-discord',
            'custom': 'fas fa-cog'
        };
        return iconMap[service.toLowerCase()] || 'fas fa-key';
    }

    updateUsageStats(stats) {
        document.getElementById('requests-today').textContent = stats.requests_today || '0';
        document.getElementById('requests-month').textContent = stats.requests_month || '0';
        document.getElementById('active-keys').textContent = stats.active_keys || '0';
        document.getElementById('failed-requests').textContent = stats.failed_requests || '0';
        
        // Update rate limit progress bars
        if (stats.hourly_limit) {
            const hourlyPercent = (stats.hourly_usage / stats.hourly_limit) * 100;
            document.getElementById('hourly-usage').textContent = `${stats.hourly_usage}/${stats.hourly_limit}`;
            document.getElementById('hourly-progress').style.width = `${Math.min(hourlyPercent, 100)}%`;
        }
        
        if (stats.daily_limit) {
            const dailyPercent = (stats.daily_usage / stats.daily_limit) * 100;
            document.getElementById('daily-usage').textContent = `${stats.daily_usage}/${stats.daily_limit}`;
            document.getElementById('daily-progress').style.width = `${Math.min(dailyPercent, 100)}%`;
        }
    }

    async generateCollection() {
        const settings = {
            name: document.getElementById('collection-name').value,
            description: document.getElementById('collection-description').value,
            version: document.getElementById('collection-version').value,
            baseUrl: document.getElementById('api-base-url').value,
            endpoints: {
                workspaces: document.getElementById('include-workspaces').checked,
                datasets: document.getElementById('include-datasets').checked,
                generation: document.getElementById('include-generation').checked,
                training: document.getElementById('include-training').checked,
                validation: document.getElementById('include-validation').checked
            }
        };

        try {
            const response = await fetch('/api/v1/postman/generate-collection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            if (!response.ok) {
                throw new Error(`Failed to generate collection: ${response.statusText}`);
            }

            const collection = await response.json();
            
            // Download the collection
            this.downloadCollection(collection);
            
            this.enhancedFeatures.showNotification('success', 'Collection Generated', 'Postman collection downloaded successfully');
            
        } catch (error) {
            console.error('Failed to generate collection:', error);
            this.enhancedFeatures.showNotification('error', 'Generation Failed', error.message);
        }
    }

    downloadCollection(collection) {
        const blob = new Blob([JSON.stringify(collection, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${collection.info.name}.postman_collection.json`;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    async runQuickTest() {
        const method = document.getElementById('test-method').value;
        const endpoint = document.getElementById('test-endpoint').value;
        const resultsContainer = document.getElementById('test-results');
        
        if (!endpoint) {
            this.enhancedFeatures.showNotification('warning', 'Missing Endpoint', 'Please provide an endpoint to test');
            return;
        }

        const baseUrl = document.getElementById('api-base-url').value;
        const fullUrl = `${baseUrl}${endpoint}`;
        
        resultsContainer.textContent = 'Running test...';

        try {
            const startTime = Date.now();
            const response = await fetch(fullUrl, { method });
            const endTime = Date.now();
            
            const responseTime = endTime - startTime;
            const responseText = await response.text();
            
            let responseData;
            try {
                responseData = JSON.parse(responseText);
            } catch {
                responseData = responseText;
            }

            const results = {
                status: response.status,
                statusText: response.statusText,
                responseTime: `${responseTime}ms`,
                headers: Object.fromEntries(response.headers.entries()),
                data: responseData
            };

            resultsContainer.textContent = JSON.stringify(results, null, 2);
            
            if (response.ok) {
                this.enhancedFeatures.showNotification('success', 'Test Passed', `${method} ${endpoint} - ${response.status}`);
            } else {
                this.enhancedFeatures.showNotification('warning', 'Test Failed', `${method} ${endpoint} - ${response.status}`);
            }

        } catch (error) {
            const errorResult = {
                error: error.message,
                endpoint: fullUrl,
                method: method
            };
            
            resultsContainer.textContent = JSON.stringify(errorResult, null, 2);
            this.enhancedFeatures.showNotification('error', 'Test Error', error.message);
        }
    }

    openCreateAPIKeyModal() {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.id = 'create-api-key-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">Create API Key</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="create-api-key-form">
                        <div class="form-group">
                            <label class="form-label required">Key Name</label>
                            <input type="text" class="form-input" name="name" required placeholder="My API Key">
                        </div>
                        <div class="form-group">
                            <label class="form-label required">Service</label>
                            <select class="form-select" name="service" required>
                                <option value="">Select a service</option>
                                <option value="postman">Postman</option>
                                <option value="google">Google APIs</option>
                                <option value="aws">AWS</option>
                                <option value="azure">Microsoft Azure</option>
                                <option value="github">GitHub</option>
                                <option value="slack">Slack</option>
                                <option value="discord">Discord</option>
                                <option value="custom">Custom/Other</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">API Key Value</label>
                            <input type="password" class="form-input" name="key_value" placeholder="Enter your API key">
                            <div class="form-help">Leave empty to generate a new key for OpenSynthetics API</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Expiration</label>
                            <select class="form-select" name="expiration">
                                <option value="">Never expires</option>
                                <option value="30">30 days</option>
                                <option value="90">90 days</option>
                                <option value="365">1 year</option>
                                <option value="custom">Custom date</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Rate Limit (requests/hour)</label>
                            <input type="number" class="form-input" name="rate_limit" placeholder="1000">
                            <div class="form-help">Leave empty for unlimited</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Scope</label>
                            <div style="margin-top: var(--spacing-sm);">
                                <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                    <input type="checkbox" name="scope" value="read" checked> Read access
                                </label>
                                <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                    <input type="checkbox" name="scope" value="write"> Write access
                                </label>
                                <label style="display: flex; align-items: center; gap: var(--spacing-sm); margin-bottom: var(--spacing-sm);">
                                    <input type="checkbox" name="scope" value="admin"> Admin access
                                </label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Description</label>
                            <textarea class="form-textarea" name="description" rows="3" placeholder="Optional description"></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                    <button class="btn btn-primary" onclick="postmanIntegration.saveAPIKey()">Create API Key</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async saveAPIKey() {
        const form = document.getElementById('create-api-key-form');
        const formData = new FormData(form);
        
        const keyData = {
            name: formData.get('name'),
            service: formData.get('service'),
            key_value: formData.get('key_value'),
            expiration: formData.get('expiration'),
            rate_limit: formData.get('rate_limit') ? parseInt(formData.get('rate_limit')) : null,
            scope: formData.getAll('scope'),
            description: formData.get('description')
        };

        try {
            const response = await fetch('/api/v1/api-keys', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(keyData)
            });

            if (!response.ok) {
                throw new Error(`Failed to create API key: ${response.statusText}`);
            }

            const result = await response.json();
            
            document.getElementById('create-api-key-modal').remove();
            this.enhancedFeatures.showNotification('success', 'API Key Created', `Successfully created API key: ${keyData.name}`);
            
            this.loadAPIKeys();

        } catch (error) {
            console.error('Failed to create API key:', error);
            this.enhancedFeatures.showNotification('error', 'Creation Failed', error.message);
        }
    }

    async copyAPIKey(keyId) {
        try {
            const response = await fetch(`/api/v1/api-keys/${keyId}/reveal`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to retrieve API key');
            }
            
            const data = await response.json();
            await navigator.clipboard.writeText(data.key_value);
            
            this.enhancedFeatures.showNotification('success', 'Copied', 'API key copied to clipboard');
            
        } catch (error) {
            console.error('Failed to copy API key:', error);
            this.enhancedFeatures.showNotification('error', 'Copy Failed', error.message);
        }
    }

    async revealAPIKey(keyId) {
        try {
            const response = await fetch(`/api/v1/api-keys/${keyId}/reveal`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to retrieve API key');
            }
            
            const data = await response.json();
            const keyElement = document.getElementById(`key-value-${keyId}`);
            
            if (keyElement) {
                keyElement.textContent = data.key_value;
                
                // Hide it again after 10 seconds
                setTimeout(() => {
                    keyElement.textContent = data.masked_key;
                }, 10000);
            }
            
        } catch (error) {
            console.error('Failed to reveal API key:', error);
            this.enhancedFeatures.showNotification('error', 'Reveal Failed', error.message);
        }
    }

    toggleAPIKeyActions(keyId) {
        const actionsElement = document.getElementById(`actions-${keyId}`);
        if (actionsElement) {
            actionsElement.classList.toggle('hidden');
        }
    }

    async revokeAPIKey(keyId) {
        if (!confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`/api/v1/api-keys/${keyId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`Failed to revoke API key: ${response.statusText}`);
            }

            this.enhancedFeatures.showNotification('success', 'API Key Revoked', 'API key has been successfully revoked');
            this.loadAPIKeys();

        } catch (error) {
            console.error('Failed to revoke API key:', error);
            this.enhancedFeatures.showNotification('error', 'Revocation Failed', error.message);
        }
    }

    refreshAPIKeys() {
        this.loadAPIKeys();
    }

    refreshCollections() {
        this.loadCollections();
    }

    async loadCollections() {
        // This would load existing collections
        const container = document.getElementById('collections-list');
        container.innerHTML = `
            <div class="text-center">
                <i class="fas fa-layer-group fa-3x" style="color: var(--text-tertiary); margin-bottom: var(--spacing-lg);"></i>
                <h3>No collections found</h3>
                <p style="color: var(--text-secondary);">Generate your first Postman collection to get started</p>
            </div>
        `;
    }

    // Additional methods for Postman workspace integration...
    connectPostman() {
        this.enhancedFeatures.showNotification('info', 'Postman Integration', 'Postman workspace integration will be implemented with OAuth flow');
    }

    showPostmanConfig() {
        this.enhancedFeatures.showNotification('info', 'Configuration', 'Postman API configuration modal will be implemented');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (window.enhancedFeatures) {
        window.postmanIntegration = new PostmanIntegration(window.enhancedFeatures);
    }
}); 