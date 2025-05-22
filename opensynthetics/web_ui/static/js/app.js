/**
 * OpenSynthetics Web UI JavaScript
 * Provides functionality for the synthetic data generation platform
 */

class OpenSyntheticsApp {
    constructor() {
        this.apiBaseUrl = '/api/v1';
        this.currentPage = 'dashboard';
        this.workspaces = [];
        this.datasets = [];
        this.strategies = {};
        
        this.init();
    }

    init() {
        // Initialize the application
        this.loadDashboardData();
        this.loadWorkspaces();
        this.loadStrategies();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.refreshDashboard();
    }

    setupEventListeners() {
        // Handle clicks outside modals to close them
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target.id);
            }
        });

        // Handle escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const activeModal = document.querySelector('.modal.active');
                if (activeModal) {
                    this.closeModal(activeModal.id);
                }
            }
        });
    }

    async apiCall(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            this.showError(`API call failed: ${error.message}`);
            throw error;
        }
    }

    showError(message) {
        // Remove existing error messages
        const existingErrors = document.querySelectorAll('.error-message');
        existingErrors.forEach(el => el.remove());
        
        // Create new error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        
        // Insert at the top of main content
        const mainContent = document.querySelector('.main-content');
        mainContent.insertBefore(errorDiv, mainContent.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }

    showSuccess(message) {
        // Remove existing success messages
        const existingSuccess = document.querySelectorAll('.success-message');
        existingSuccess.forEach(el => el.remove());
        
        // Create new success message
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        
        // Insert at the top of main content
        const mainContent = document.querySelector('.main-content');
        mainContent.insertBefore(successDiv, mainContent.firstChild);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.remove();
            }
        }, 3000);
    }

    showPage(pageId) {
        // Hide all pages
        const pages = document.querySelectorAll('.page');
        pages.forEach(page => page.classList.add('hidden'));
        
        // Show selected page
        const targetPage = document.getElementById(`${pageId}-page`);
        if (targetPage) {
            targetPage.classList.remove('hidden');
        }
        
        // Update navigation
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => item.classList.remove('active'));
        
        const activeNavItem = document.querySelector(`[onclick="showPage('${pageId}')"]`);
        if (activeNavItem) {
            activeNavItem.classList.add('active');
        }
        
        this.currentPage = pageId;
        
        // Load page-specific data
        switch (pageId) {
            case 'dashboard':
                this.refreshDashboard();
                break;
            case 'workspaces':
                this.loadWorkspacesList();
                break;
            case 'generate':
                this.loadGeneratePage();
                break;
            case 'datasets':
                this.loadDatasetsList();
                break;
        }
    }

    async refreshDashboard() {
        try {
            // Load workspace count
            const workspaces = await this.apiCall('/workspaces');
            document.getElementById('total-workspaces').textContent = workspaces.data?.length || 0;
            
            // Load dataset count (sum across all workspaces)
            let totalDatasets = 0;
            let totalRecords = 0;
            
            if (workspaces.data) {
                for (const workspace of workspaces.data) {
                    try {
                        const datasets = await this.apiCall(`/workspaces/${encodeURIComponent(workspace.path)}/datasets`);
                        if (datasets.data) {
                            totalDatasets += datasets.data.length;
                            // Sum up records from all datasets
                            datasets.data.forEach(dataset => {
                                totalRecords += dataset.record_count || 0;
                            });
                        }
                    } catch (error) {
                        console.warn(`Failed to load datasets for workspace ${workspace.name}:`, error);
                    }
                }
            }
            
            document.getElementById('total-datasets').textContent = totalDatasets;
            document.getElementById('total-records').textContent = totalRecords.toLocaleString();
            
            // Load recent activities
            this.loadRecentActivities();
        } catch (error) {
            console.error('Failed to refresh dashboard:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    async loadRecentActivities() {
        const activitiesContainer = document.getElementById('recent-activities');
        activitiesContainer.innerHTML = '<p>No recent activities found.</p>';
        
        // This would typically load from an activities API endpoint
        // For now, we'll show a placeholder
        activitiesContainer.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="padding: 12px; background-color: var(--background-color); border-radius: 8px;">
                    <div style="font-weight: 500;">Dataset Created</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">mechanical_problems - 2 hours ago</div>
                </div>
                <div style="padding: 12px; background-color: var(--background-color); border-radius: 8px;">
                    <div style="font-weight: 500;">Workspace Created</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">engineering_project - 1 day ago</div>
                </div>
            </div>
        `;
    }

    async loadWorkspaces() {
        try {
            const response = await this.apiCall('/workspaces');
            this.workspaces = response.data || [];
        } catch (error) {
            console.error('Failed to load workspaces:', error);
            this.workspaces = [];
        }
    }

    async loadWorkspacesList() {
        const container = document.getElementById('workspaces-list');
        container.innerHTML = '<div class="loading"></div> Loading workspaces...';
        
        try {
            await this.loadWorkspaces();
            
            if (this.workspaces.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; padding: 40px;">
                        <i class="fas fa-folder-open" style="font-size: 48px; color: var(--text-secondary); margin-bottom: 16px;"></i>
                        <h3>No workspaces found</h3>
                        <p style="color: var(--text-secondary); margin-bottom: 20px;">Create your first workspace to get started.</p>
                        <button class="btn btn-primary" onclick="app.openNewWorkspaceModal()">
                            <i class="fas fa-plus"></i>
                            Create Workspace
                        </button>
                    </div>
                `;
                return;
            }
            
            let html = '<div class="grid grid-2">';
            
            for (const workspace of this.workspaces) {
                html += `
                    <div class="card">
                        <h4 style="margin-bottom: 8px;">${workspace.name}</h4>
                        <p style="color: var(--text-secondary); margin-bottom: 16px;">${workspace.description || 'No description'}</p>
                        <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                            ${workspace.tags ? workspace.tags.map(tag => `<span class="status-badge status-active">${tag}</span>`).join('') : ''}
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn btn-primary" onclick="app.openWorkspace('${workspace.path}')">
                                <i class="fas fa-folder-open"></i>
                                Open
                            </button>
                            <button class="btn btn-secondary" onclick="app.deleteWorkspace('${workspace.path}')">
                                <i class="fas fa-trash"></i>
                                Delete
                            </button>
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            container.innerHTML = html;
        } catch (error) {
            container.innerHTML = '<p>Failed to load workspaces.</p>';
        }
    }

    async loadStrategies() {
        try {
            const response = await this.apiCall('/strategies');
            this.strategies = response.data || {};
        } catch (error) {
            console.error('Failed to load strategies:', error);
            this.strategies = {};
        }
    }

    async loadGeneratePage() {
        // Populate workspace select
        const workspaceSelect = document.getElementById('workspace-select');
        workspaceSelect.innerHTML = '<option value="">Select a workspace...</option>';
        
        this.workspaces.forEach(workspace => {
            const option = document.createElement('option');
            option.value = workspace.path;
            option.textContent = workspace.name;
            workspaceSelect.appendChild(option);
        });
        
        // Populate strategy select
        const strategySelect = document.getElementById('strategy-select');
        strategySelect.innerHTML = '<option value="">Select a strategy...</option>';
        
        Object.keys(this.strategies).forEach(strategyName => {
            const option = document.createElement('option');
            option.value = strategyName;
            option.textContent = this.strategies[strategyName].name || strategyName;
            strategySelect.appendChild(option);
        });
    }

    updateStrategyForm() {
        const strategySelect = document.getElementById('strategy-select');
        const parametersContainer = document.getElementById('strategy-parameters');
        const selectedStrategy = strategySelect.value;
        
        if (!selectedStrategy || !this.strategies[selectedStrategy]) {
            parametersContainer.innerHTML = '';
            return;
        }
        
        const strategy = this.strategies[selectedStrategy];
        parametersContainer.innerHTML = this.generateStrategyForm(selectedStrategy, strategy);
    }

    generateStrategyForm(strategyName, strategy) {
        if (strategyName === 'engineering_problems') {
            return `
                <div class="grid grid-2">
                    <div class="form-group">
                        <label class="form-label">Domain</label>
                        <select class="form-select" id="param-domain" required>
                            <option value="mechanical">Mechanical</option>
                            <option value="electrical">Electrical</option>
                            <option value="civil">Civil</option>
                            <option value="chemical">Chemical</option>
                            <option value="software">Software</option>
                            <option value="aerospace">Aerospace</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Count</label>
                        <input type="number" class="form-input" id="param-count" value="5" min="1" max="100" required>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Difficulty (1-10)</label>
                    <input type="range" class="form-input" id="param-difficulty" value="5" min="1" max="10" 
                           oninput="document.getElementById('difficulty-value').textContent = this.value">
                    <div style="text-align: center; margin-top: 8px;">
                        Difficulty: <span id="difficulty-value">5</span>/10
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Constraints (Optional)</label>
                    <textarea class="form-textarea" id="param-constraints" 
                              placeholder="Enter any specific constraints or requirements..."></textarea>
                </div>
            `;
        } else if (strategyName === 'system_design') {
            return `
                <div class="form-group">
                    <label class="form-label">Requirements</label>
                    <textarea class="form-textarea" id="param-requirements" required
                              placeholder="Describe the system requirements..."></textarea>
                </div>
                <div class="form-group">
                    <label class="form-label">Constraints</label>
                    <textarea class="form-textarea" id="param-constraints" required
                              placeholder="Describe the design constraints..."></textarea>
                </div>
            `;
        }
        
        return '<p>No parameters required for this strategy.</p>';
    }

    async generateData(event) {
        event.preventDefault();
        
        const workspacePath = document.getElementById('workspace-select').value;
        const strategy = document.getElementById('strategy-select').value;
        const datasetName = document.getElementById('dataset-name').value;
        
        if (!workspacePath || !strategy || !datasetName) {
            this.showError('Please fill in all required fields.');
            return;
        }
        
        // Collect strategy parameters
        const parameters = this.collectStrategyParameters(strategy);
        
        // Show generation status
        const statusCard = document.getElementById('generation-status');
        const progressContainer = document.getElementById('generation-progress');
        statusCard.classList.remove('hidden');
        progressContainer.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="loading"></div>
                <span>Generating data...</span>
            </div>
        `;
        
        try {
            const response = await this.apiCall('/generate/jobs', {
                method: 'POST',
                body: JSON.stringify({
                    workspace_path: workspacePath,
                    strategy: strategy,
                    parameters: parameters,
                    output_dataset: datasetName
                })
            });
            
            if (response.data) {
                progressContainer.innerHTML = `
                    <div class="success-message">
                        <i class="fas fa-check-circle"></i>
                        Successfully generated ${response.data.count} items in dataset '${datasetName}'
                    </div>
                    ${response.data.sample_items ? this.renderSampleItems(response.data.sample_items) : ''}
                `;
                this.showSuccess('Data generation completed successfully!');
                
                // Reset form
                document.getElementById('generation-form').reset();
                document.getElementById('strategy-parameters').innerHTML = '';
            }
        } catch (error) {
            progressContainer.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    Generation failed: ${error.message}
                </div>
            `;
        }
    }

    collectStrategyParameters(strategy) {
        const parameters = {};
        
        if (strategy === 'engineering_problems') {
            parameters.domain = document.getElementById('param-domain').value;
            parameters.count = parseInt(document.getElementById('param-count').value);
            parameters.difficulty = parseInt(document.getElementById('param-difficulty').value);
            parameters.constraints = document.getElementById('param-constraints').value;
        } else if (strategy === 'system_design') {
            parameters.requirements = document.getElementById('param-requirements').value;
            parameters.constraints = document.getElementById('param-constraints').value;
        }
        
        return parameters;
    }

    renderSampleItems(items) {
        if (!items || items.length === 0) return '';
        
        let html = '<h4 style="margin: 20px 0 12px 0;">Sample Generated Items:</h4>';
        html += '<div style="max-height: 300px; overflow-y: auto;">';
        
        items.forEach((item, index) => {
            html += `
                <div style="padding: 12px; background-color: var(--background-color); border-radius: 8px; margin-bottom: 8px;">
                    <h5>Item ${index + 1}</h5>
                    <pre style="white-space: pre-wrap; font-size: 0.9rem; margin: 8px 0 0 0;">${JSON.stringify(item, null, 2)}</pre>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }

    async loadDatasetsList() {
        const container = document.getElementById('datasets-list');
        container.innerHTML = '<div class="loading"></div> Loading datasets...';
        
        try {
            const datasets = [];
            
            // Load datasets from all workspaces
            for (const workspace of this.workspaces) {
                try {
                    const response = await this.apiCall(`/workspaces/${encodeURIComponent(workspace.path)}/datasets`);
                    if (response.data) {
                        response.data.forEach(dataset => {
                            datasets.push({
                                ...dataset,
                                workspace_name: workspace.name,
                                workspace_path: workspace.path
                            });
                        });
                    }
                } catch (error) {
                    console.warn(`Failed to load datasets for workspace ${workspace.name}:`, error);
                }
            }
            
            if (datasets.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; padding: 40px;">
                        <i class="fas fa-table" style="font-size: 48px; color: var(--text-secondary); margin-bottom: 16px;"></i>
                        <h3>No datasets found</h3>
                        <p style="color: var(--text-secondary); margin-bottom: 20px;">Generate your first dataset to get started.</p>
                        <button class="btn btn-primary" onclick="app.showPage('generate')">
                            <i class="fas fa-magic"></i>
                            Generate Data
                        </button>
                    </div>
                `;
                return;
            }
            
            // Create table
            let html = `
                <table class="table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Workspace</th>
                            <th>Description</th>
                            <th>Records</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            datasets.forEach(dataset => {
                html += `
                    <tr>
                        <td><strong>${dataset.name}</strong></td>
                        <td>${dataset.workspace_name}</td>
                        <td>${dataset.description || '-'}</td>
                        <td>${dataset.record_count || 0}</td>
                        <td>${new Date(dataset.created_at).toLocaleDateString()}</td>
                        <td>
                            <button class="btn btn-secondary" onclick="app.viewDataset('${dataset.workspace_path}', '${dataset.name}')">
                                <i class="fas fa-eye"></i>
                                View
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        } catch (error) {
            container.innerHTML = '<p>Failed to load datasets.</p>';
        }
    }

    // Modal functions
    openNewWorkspaceModal() {
        const modal = document.getElementById('new-workspace-modal');
        modal.classList.add('active');
    }

    openSettingsModal() {
        const modal = document.getElementById('settings-modal');
        modal.classList.add('active');
        
        // Load current settings
        this.loadCurrentSettings();
    }

    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.remove('active');
    }

    async createWorkspace(event) {
        event.preventDefault();
        
        const name = document.getElementById('new-workspace-name').value;
        const description = document.getElementById('new-workspace-description').value;
        const tagsString = document.getElementById('new-workspace-tags').value;
        const tags = tagsString ? tagsString.split(',').map(tag => tag.trim()) : [];
        
        try {
            const response = await this.apiCall('/workspaces', {
                method: 'POST',
                body: JSON.stringify({
                    name: name,
                    description: description,
                    tags: tags
                })
            });
            
            if (response.data) {
                this.showSuccess(`Workspace '${name}' created successfully!`);
                this.closeModal('new-workspace-modal');
                
                // Refresh workspaces list
                await this.loadWorkspaces();
                
                // If we're on the workspaces page, refresh the list
                if (this.currentPage === 'workspaces') {
                    this.loadWorkspacesList();
                }
                
                // Reset form
                document.getElementById('new-workspace-form').reset();
            }
        } catch (error) {
            this.showError(`Failed to create workspace: ${error.message}`);
        }
    }

    async saveSettings(event) {
        event.preventDefault();
        
        const openaiKey = document.getElementById('openai-api-key').value;
        const defaultModel = document.getElementById('default-model').value;
        
        try {
            if (openaiKey) {
                await this.apiCall('/config/api_keys/openai', {
                    method: 'POST',
                    body: JSON.stringify({ key: openaiKey })
                });
            }
            
            // Save other settings as needed
            this.showSuccess('Settings saved successfully!');
            this.closeModal('settings-modal');
        } catch (error) {
            this.showError(`Failed to save settings: ${error.message}`);
        }
    }

    async loadCurrentSettings() {
        try {
            const response = await this.apiCall('/config');
            if (response.data) {
                // Populate settings form with current values
                // Note: Don't show API keys for security
                if (response.data.llm && response.data.llm.default_model) {
                    document.getElementById('default-model').value = response.data.llm.default_model;
                }
            }
        } catch (error) {
            console.warn('Failed to load current settings:', error);
        }
    }

    async openWorkspace(workspacePath) {
        // This could navigate to a detailed workspace view
        this.showSuccess(`Opening workspace: ${workspacePath}`);
        // For now, just show the datasets page filtered to this workspace
        this.showPage('datasets');
    }

    async deleteWorkspace(workspacePath) {
        if (!confirm('Are you sure you want to delete this workspace? This action cannot be undone.')) {
            return;
        }
        
        try {
            await this.apiCall(`/workspaces/${encodeURIComponent(workspacePath)}`, {
                method: 'DELETE'
            });
            
            this.showSuccess('Workspace deleted successfully!');
            await this.loadWorkspaces();
            this.loadWorkspacesList();
        } catch (error) {
            this.showError(`Failed to delete workspace: ${error.message}`);
        }
    }

    async viewDataset(workspacePath, datasetName) {
        // This could open a detailed dataset view
        this.showSuccess(`Viewing dataset: ${datasetName}`);
        // For now, just show a basic info message
    }

    async loadDashboardData() {
        // Load initial dashboard data
        await this.loadWorkspaces();
        await this.loadStrategies();
    }
}

// Global functions for HTML onclick handlers
function showPage(pageId) {
    app.showPage(pageId);
}

function openNewWorkspaceModal() {
    app.openNewWorkspaceModal();
}

function openSettingsModal() {
    app.openSettingsModal();
}

function closeModal(modalId) {
    app.closeModal(modalId);
}

function createWorkspace(event) {
    app.createWorkspace(event);
}

function saveSettings(event) {
    app.saveSettings(event);
}

function generateData(event) {
    app.generateData(event);
}

function updateStrategyForm() {
    app.updateStrategyForm();
}

function loadDatasets() {
    app.loadDatasetsList();
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OpenSyntheticsApp();
}); 