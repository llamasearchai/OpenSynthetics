/**
 * OpenSynthetics Enhanced Features
 * File Upload, Cloud Storage, MCP Servers, and Advanced UI Components
 */

class EnhancedFeatures {
    constructor(app) {
        this.app = app;
        this.fileUploads = new Map();
        this.cloudProviders = new Map();
        this.mcpServers = new Map();
        this.notifications = [];
        
        this.init();
    }

    init() {
        this.initFileUpload();
        this.initCloudStorage();
        this.initMCPServers();
        this.initNotifications();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Global drag and drop
        document.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        document.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Only handle if not in a specific drop zone
            if (!e.target.closest('.file-upload-area')) {
                this.handleGlobalFileDrop(e);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'u':
                        e.preventDefault();
                        this.openFileUploadModal();
                        break;
                    case 'n':
                        e.preventDefault();
                        this.app.openNewWorkspaceModal();
                        break;
                    case 'g':
                        e.preventDefault();
                        this.app.showPage('generate');
                        break;
                }
            }
        });
    }

    // File Upload System
    initFileUpload() {
        this.createFileUploadModal();
    }

    createFileUploadModal() {
        const modal = document.createElement('div');
        modal.id = 'file-upload-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 800px;">
                <div class="modal-header">
                    <h2 class="modal-title">Upload Files</h2>
                    <button class="modal-close" onclick="closeModal('file-upload-modal')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="file-upload">
                        <input type="file" id="file-upload-input" class="file-upload-input" multiple>
                        <div class="file-upload-area" onclick="document.getElementById('file-upload-input').click()">
                            <div class="file-upload-icon">
                                <i class="fas fa-cloud-upload-alt"></i>
                            </div>
                            <div class="file-upload-text">Click to upload files or drag and drop</div>
                            <div class="file-upload-hint">Supports: CSV, JSON, XLSX, PDF, TXT (Max: 100MB per file)</div>
                        </div>
                        <div class="file-upload-progress">
                            <div class="progress-bar">
                                <div class="progress-fill"></div>
                            </div>
                            <div class="progress-text">Uploading...</div>
                        </div>
                    </div>
                    
                    <div class="file-list" id="upload-file-list"></div>
                    
                    <div class="cloud-storage">
                        <div class="cloud-provider" onclick="enhancedFeatures.connectGoogleDrive()">
                            <div class="cloud-provider-icon" style="color: #4285F4;">
                                <i class="fab fa-google-drive"></i>
                            </div>
                            <div class="cloud-provider-name">Google Drive</div>
                            <div class="cloud-provider-status" id="gdrive-status">Not connected</div>
                        </div>
                        
                        <div class="cloud-provider" onclick="enhancedFeatures.connectDropbox()">
                            <div class="cloud-provider-icon" style="color: #0061FF;">
                                <i class="fab fa-dropbox"></i>
                            </div>
                            <div class="cloud-provider-name">Dropbox</div>
                            <div class="cloud-provider-status" id="dropbox-status">Not connected</div>
                        </div>
                        
                        <div class="cloud-provider" onclick="enhancedFeatures.connectOneDrive()">
                            <div class="cloud-provider-icon" style="color: #0078D4;">
                                <i class="fab fa-microsoft"></i>
                            </div>
                            <div class="cloud-provider-name">OneDrive</div>
                            <div class="cloud-provider-status" id="onedrive-status">Not connected</div>
                        </div>
                        
                        <div class="cloud-provider" onclick="enhancedFeatures.connectAWS()">
                            <div class="cloud-provider-icon" style="color: #FF9900;">
                                <i class="fab fa-aws"></i>
                            </div>
                            <div class="cloud-provider-name">AWS S3</div>
                            <div class="cloud-provider-status" id="aws-status">Not connected</div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeModal('file-upload-modal')">Cancel</button>
                    <button class="btn btn-primary" onclick="enhancedFeatures.processUploadedFiles()">Process Files</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        this.setupFileUploadEvents();
    }

    setupFileUploadEvents() {
        const input = document.getElementById('file-upload-input');
        const area = document.querySelector('.file-upload-area');

        input.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });

        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });

        area.addEventListener('dragleave', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
        });

        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            this.handleFileSelection(e.dataTransfer.files);
        });
    }

    handleFileSelection(files) {
        const fileList = document.getElementById('upload-file-list');
        
        Array.from(files).forEach(file => {
            if (this.validateFile(file)) {
                this.addFileToList(file, fileList);
            }
        });
    }

    validateFile(file) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = [
            'text/csv',
            'application/json',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/pdf',
            'text/plain'
        ];

        if (file.size > maxSize) {
            this.showNotification('error', 'File too large', `${file.name} exceeds 100MB limit`);
            return false;
        }

        if (!allowedTypes.includes(file.type)) {
            this.showNotification('warning', 'Unsupported file type', `${file.name} is not a supported file type`);
            return false;
        }

        return true;
    }

    addFileToList(file, container) {
        const fileId = this.generateId();
        this.fileUploads.set(fileId, file);

        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-icon">
                <i class="${this.getFileIcon(file.type)}"></i>
            </div>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${this.formatFileSize(file.size)}</div>
            </div>
            <div class="file-actions">
                <button class="btn btn-sm btn-secondary" onclick="enhancedFeatures.previewFile('${fileId}')">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="enhancedFeatures.removeFile('${fileId}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        container.appendChild(fileItem);
    }

    getFileIcon(mimeType) {
        const iconMap = {
            'text/csv': 'fas fa-file-csv',
            'application/json': 'fas fa-file-code',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'fas fa-file-excel',
            'application/pdf': 'fas fa-file-pdf',
            'text/plain': 'fas fa-file-alt'
        };
        return iconMap[mimeType] || 'fas fa-file';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async previewFile(fileId) {
        const file = this.fileUploads.get(fileId);
        if (!file) return;

        const modal = this.createPreviewModal(file);
        document.body.appendChild(modal);
        modal.classList.add('active');

        if (file.type === 'text/csv' || file.type === 'application/json' || file.type === 'text/plain') {
            const text = await file.text();
            this.displayTextPreview(text, file.type);
        } else if (file.type === 'application/pdf') {
            this.displayPDFPreview(file);
        }
    }

    createPreviewModal(file) {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.id = 'file-preview-modal';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 90vw; max-height: 90vh;">
                <div class="modal-header">
                    <h2 class="modal-title">Preview: ${file.name}</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body" id="preview-content">
                    <div class="text-center">
                        <i class="fas fa-spinner fa-spin fa-2x"></i>
                        <p>Loading preview...</p>
                    </div>
                </div>
            </div>
        `;
        return modal;
    }

    displayTextPreview(text, type) {
        const content = document.getElementById('preview-content');
        
        if (type === 'application/json') {
            try {
                const parsed = JSON.parse(text);
                text = JSON.stringify(parsed, null, 2);
            } catch (e) {
                // Keep original text if JSON parsing fails
            }
        }

        content.innerHTML = `
            <pre style="
                background-color: var(--surface-secondary);
                padding: var(--spacing-lg);
                border-radius: var(--border-radius);
                overflow: auto;
                max-height: 60vh;
                font-family: var(--font-mono);
                font-size: 0.9rem;
                white-space: pre-wrap;
            ">${text.substring(0, 10000)}${text.length > 10000 ? '\n\n... (truncated)' : ''}</pre>
        `;
    }

    removeFile(fileId) {
        this.fileUploads.delete(fileId);
        const fileItem = event.target.closest('.file-item');
        if (fileItem) {
            fileItem.remove();
        }
    }

    async processUploadedFiles() {
        if (this.fileUploads.size === 0) {
            this.showNotification('warning', 'No files selected', 'Please select files to upload');
            return;
        }

        const progressContainer = document.querySelector('.file-upload-progress');
        const progressFill = document.querySelector('.progress-fill');
        const progressText = document.querySelector('.progress-text');
        
        progressContainer.style.display = 'block';

        let processed = 0;
        const total = this.fileUploads.size;

        for (const [fileId, file] of this.fileUploads) {
            try {
                await this.uploadFile(file);
                processed++;
                
                const progress = (processed / total) * 100;
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `Processing ${processed}/${total} files...`;
                
            } catch (error) {
                this.showNotification('error', 'Upload failed', `Failed to upload ${file.name}: ${error.message}`);
            }
        }

        progressText.textContent = 'Upload complete!';
        setTimeout(() => {
            this.closeModal('file-upload-modal');
            this.showNotification('success', 'Files uploaded', `Successfully processed ${processed} files`);
        }, 1000);
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/v1/files/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        return await response.json();
    }

    openFileUploadModal() {
        const modal = document.getElementById('file-upload-modal');
        if (modal) {
            modal.classList.add('active');
        }
    }

    handleGlobalFileDrop(e) {
        if (e.dataTransfer.files.length > 0) {
            this.openFileUploadModal();
            setTimeout(() => {
                this.handleFileSelection(e.dataTransfer.files);
            }, 100);
        }
    }

    // Cloud Storage Integration
    initCloudStorage() {
        this.loadCloudProviderStatus();
    }

    async loadCloudProviderStatus() {
        try {
            const response = await fetch('/api/v1/cloud/status');
            const data = await response.json();
            
            Object.keys(data).forEach(provider => {
                this.updateProviderStatus(provider, data[provider]);
            });
        } catch (error) {
            console.warn('Failed to load cloud provider status:', error);
        }
    }

    updateProviderStatus(provider, status) {
        const providerElement = document.getElementById(`${provider}-status`);
        const providerCard = providerElement?.closest('.cloud-provider');
        
        if (providerElement && providerCard) {
            if (status.connected) {
                providerElement.textContent = 'Connected';
                providerCard.classList.add('connected');
            } else {
                providerElement.textContent = 'Not connected';
                providerCard.classList.remove('connected');
            }
        }
    }

    async connectGoogleDrive() {
        try {
            // Load Google Drive API
            if (!window.gapi) {
                await this.loadScript('https://apis.google.com/js/api.js');
            }

            await new Promise((resolve) => {
                gapi.load('auth2:picker', resolve);
            });

            const authInstance = gapi.auth2.getAuthInstance();
            if (!authInstance) {
                await gapi.auth2.init({
                    client_id: this.getGoogleClientId(),
                    scope: 'https://www.googleapis.com/auth/drive.readonly'
                });
            }

            const user = await gapi.auth2.getAuthInstance().signIn();
            const accessToken = user.getAuthResponse().access_token;

            // Store the access token
            await this.storeCloudCredentials('google_drive', {
                access_token: accessToken,
                user_info: user.getBasicProfile()
            });

            this.updateProviderStatus('gdrive', { connected: true });
            this.showNotification('success', 'Google Drive connected', 'Successfully connected to Google Drive');

        } catch (error) {
            this.showNotification('error', 'Connection failed', `Failed to connect to Google Drive: ${error.message}`);
        }
    }

    async connectDropbox() {
        try {
            // Dropbox OAuth flow
            const clientId = this.getDropboxClientId();
            const redirectUri = `${window.location.origin}/auth/dropbox/callback`;
            
            const authUrl = `https://www.dropbox.com/oauth2/authorize?client_id=${clientId}&response_type=code&redirect_uri=${encodeURIComponent(redirectUri)}`;
            
            const popup = window.open(authUrl, 'dropbox_auth', 'width=500,height=600');
            
            const accessToken = await this.waitForAuthCallback(popup, 'dropbox');
            
            await this.storeCloudCredentials('dropbox', { access_token: accessToken });
            
            this.updateProviderStatus('dropbox', { connected: true });
            this.showNotification('success', 'Dropbox connected', 'Successfully connected to Dropbox');

        } catch (error) {
            this.showNotification('error', 'Connection failed', `Failed to connect to Dropbox: ${error.message}`);
        }
    }

    async connectOneDrive() {
        try {
            // Microsoft Graph OAuth flow
            const clientId = this.getMicrosoftClientId();
            const redirectUri = `${window.location.origin}/auth/microsoft/callback`;
            
            const authUrl = `https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=${clientId}&response_type=code&redirect_uri=${encodeURIComponent(redirectUri)}&scope=https://graph.microsoft.com/Files.Read`;
            
            const popup = window.open(authUrl, 'onedrive_auth', 'width=500,height=600');
            
            const accessToken = await this.waitForAuthCallback(popup, 'onedrive');
            
            await this.storeCloudCredentials('onedrive', { access_token: accessToken });
            
            this.updateProviderStatus('onedrive', { connected: true });
            this.showNotification('success', 'OneDrive connected', 'Successfully connected to OneDrive');

        } catch (error) {
            this.showNotification('error', 'Connection failed', `Failed to connect to OneDrive: ${error.message}`);
        }
    }

    async connectAWS() {
        const modal = this.createAWSConfigModal();
        document.body.appendChild(modal);
        modal.classList.add('active');
    }

    createAWSConfigModal() {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.id = 'aws-config-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">Configure AWS S3</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="aws-config-form">
                        <div class="form-group">
                            <label class="form-label required">Access Key ID</label>
                            <input type="text" class="form-input" name="access_key_id" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label required">Secret Access Key</label>
                            <input type="password" class="form-input" name="secret_access_key" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label required">Region</label>
                            <select class="form-select" name="region" required>
                                <option value="">Select a region</option>
                                <option value="us-east-1">US East (N. Virginia)</option>
                                <option value="us-west-2">US West (Oregon)</option>
                                <option value="eu-west-1">Europe (Ireland)</option>
                                <option value="ap-southeast-1">Asia Pacific (Singapore)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Default Bucket</label>
                            <input type="text" class="form-input" name="bucket">
                            <div class="form-help">Optional: Default S3 bucket for uploads</div>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                    <button class="btn btn-primary" onclick="enhancedFeatures.saveAWSConfig()">Connect</button>
                </div>
            </div>
        `;
        return modal;
    }

    async saveAWSConfig() {
        const form = document.getElementById('aws-config-form');
        const formData = new FormData(form);
        const config = Object.fromEntries(formData);

        try {
            await this.storeCloudCredentials('aws', config);
            
            this.updateProviderStatus('aws', { connected: true });
            this.showNotification('success', 'AWS S3 connected', 'Successfully configured AWS S3');
            
            document.getElementById('aws-config-modal').remove();

        } catch (error) {
            this.showNotification('error', 'Configuration failed', `Failed to configure AWS S3: ${error.message}`);
        }
    }

    async storeCloudCredentials(provider, credentials) {
        const response = await fetch('/api/v1/cloud/credentials', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, credentials })
        });

        if (!response.ok) {
            throw new Error(`Failed to store credentials: ${response.statusText}`);
        }
    }

    // MCP Server Management
    initMCPServers() {
        this.createMCPServerModal();
        this.loadMCPServers();
    }

    createMCPServerModal() {
        const modal = document.createElement('div');
        modal.id = 'mcp-servers-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 900px;">
                <div class="modal-header">
                    <h2 class="modal-title">MCP Servers</h2>
                    <button class="modal-close" onclick="closeModal('mcp-servers-modal')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div style="margin-bottom: var(--spacing-lg);">
                        <button class="btn btn-primary" onclick="enhancedFeatures.openAddMCPServerModal()">
                            <i class="fas fa-plus"></i>
                            Add MCP Server
                        </button>
                        <button class="btn btn-secondary" onclick="enhancedFeatures.discoverMCPServers()">
                            <i class="fas fa-search"></i>
                            Discover Servers
                        </button>
                    </div>
                    
                    <div class="mcp-servers" id="mcp-servers-list">
                        <div class="text-center">
                            <i class="fas fa-spinner fa-spin fa-2x"></i>
                            <p>Loading MCP servers...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async loadMCPServers() {
        try {
            const response = await fetch('/api/v1/mcp/servers');
            const servers = await response.json();
            
            this.renderMCPServers(servers.data || []);
        } catch (error) {
            console.error('Failed to load MCP servers:', error);
            this.renderMCPServers([]);
        }
    }

    renderMCPServers(servers) {
        const container = document.getElementById('mcp-servers-list');
        
        if (servers.length === 0) {
            container.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-server fa-3x" style="color: var(--text-tertiary); margin-bottom: var(--spacing-lg);"></i>
                    <h3>No MCP servers configured</h3>
                    <p style="color: var(--text-secondary);">Add your first MCP server to get started</p>
                </div>
            `;
            return;
        }

        container.innerHTML = servers.map(server => this.createMCPServerHTML(server)).join('');
    }

    createMCPServerHTML(server) {
        return `
            <div class="mcp-server" data-server-id="${server.id}">
                <div class="mcp-server-header" onclick="enhancedFeatures.toggleMCPServer('${server.id}')">
                    <div class="mcp-server-info">
                        <div class="mcp-server-icon">
                            <i class="${server.icon || 'fas fa-server'}"></i>
                        </div>
                        <div class="mcp-server-details">
                            <h3>${server.name}</h3>
                            <div class="mcp-server-url">${server.url}</div>
                        </div>
                    </div>
                    <div class="mcp-server-status">
                        <span class="status-indicator ${server.status}"></span>
                        <span>${server.status === 'connected' ? 'Connected' : 'Disconnected'}</span>
                        <button class="btn btn-sm btn-icon" onclick="event.stopPropagation(); enhancedFeatures.testMCPConnection('${server.id}')">
                            <i class="fas fa-plug"></i>
                        </button>
                    </div>
                </div>
                <div class="mcp-server-content">
                    <div class="mcp-capabilities">
                        ${(server.capabilities || []).map(cap => `
                            <div class="mcp-capability">
                                <div class="mcp-capability-icon">
                                    <i class="${cap.icon || 'fas fa-cog'}"></i>
                                </div>
                                <div class="mcp-capability-name">${cap.name}</div>
                                <div class="mcp-capability-desc">${cap.description}</div>
                            </div>
                        `).join('')}
                    </div>
                    <div style="display: flex; gap: var(--spacing-md);">
                        <button class="btn btn-secondary" onclick="enhancedFeatures.editMCPServer('${server.id}')">
                            <i class="fas fa-edit"></i>
                            Edit
                        </button>
                        <button class="btn btn-danger" onclick="enhancedFeatures.deleteMCPServer('${server.id}')">
                            <i class="fas fa-trash"></i>
                            Delete
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    toggleMCPServer(serverId) {
        const server = document.querySelector(`[data-server-id="${serverId}"]`);
        server.classList.toggle('expanded');
    }

    openAddMCPServerModal() {
        const modal = this.createAddMCPServerModal();
        document.body.appendChild(modal);
        modal.classList.add('active');
    }

    createAddMCPServerModal() {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.id = 'add-mcp-server-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">Add MCP Server</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <form id="add-mcp-server-form">
                        <div class="form-group">
                            <label class="form-label required">Server Name</label>
                            <input type="text" class="form-input" name="name" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label required">Server URL</label>
                            <input type="url" class="form-input" name="url" placeholder="https://example.com/mcp" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Server Type</label>
                            <select class="form-select" name="type">
                                <option value="rest">REST API</option>
                                <option value="websocket">WebSocket</option>
                                <option value="sse">Server-Sent Events</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Authentication</label>
                            <select class="form-select" name="auth_type" onchange="enhancedFeatures.toggleAuthFields(this.value)">
                                <option value="none">None</option>
                                <option value="bearer">Bearer Token</option>
                                <option value="basic">Basic Auth</option>
                                <option value="api_key">API Key</option>
                            </select>
                        </div>
                        <div class="form-group" id="auth-fields" style="display: none;">
                            <div id="bearer-fields" style="display: none;">
                                <label class="form-label">Bearer Token</label>
                                <input type="password" class="form-input" name="bearer_token">
                            </div>
                            <div id="basic-fields" style="display: none;">
                                <label class="form-label">Username</label>
                                <input type="text" class="form-input" name="username">
                                <label class="form-label">Password</label>
                                <input type="password" class="form-input" name="password">
                            </div>
                            <div id="api-key-fields" style="display: none;">
                                <label class="form-label">API Key</label>
                                <input type="password" class="form-input" name="api_key">
                                <label class="form-label">Header Name</label>
                                <input type="text" class="form-input" name="api_key_header" placeholder="X-API-Key">
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Description</label>
                            <textarea class="form-textarea" name="description" rows="3"></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                    <button class="btn btn-primary" onclick="enhancedFeatures.saveMCPServer()">Add Server</button>
                </div>
            </div>
        `;
        return modal;
    }

    toggleAuthFields(authType) {
        const authFields = document.getElementById('auth-fields');
        const bearerFields = document.getElementById('bearer-fields');
        const basicFields = document.getElementById('basic-fields');
        const apiKeyFields = document.getElementById('api-key-fields');

        // Hide all fields first
        authFields.style.display = 'none';
        bearerFields.style.display = 'none';
        basicFields.style.display = 'none';
        apiKeyFields.style.display = 'none';

        if (authType !== 'none') {
            authFields.style.display = 'block';
            
            switch (authType) {
                case 'bearer':
                    bearerFields.style.display = 'block';
                    break;
                case 'basic':
                    basicFields.style.display = 'block';
                    break;
                case 'api_key':
                    apiKeyFields.style.display = 'block';
                    break;
            }
        }
    }

    async saveMCPServer() {
        const form = document.getElementById('add-mcp-server-form');
        const formData = new FormData(form);
        const serverData = Object.fromEntries(formData);

        try {
            const response = await fetch('/api/v1/mcp/servers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(serverData)
            });

            if (!response.ok) {
                throw new Error(`Failed to add server: ${response.statusText}`);
            }

            const result = await response.json();
            
            document.getElementById('add-mcp-server-modal').remove();
            this.loadMCPServers();
            this.showNotification('success', 'Server added', `Successfully added MCP server: ${serverData.name}`);

        } catch (error) {
            this.showNotification('error', 'Failed to add server', error.message);
        }
    }

    async testMCPConnection(serverId) {
        try {
            const response = await fetch(`/api/v1/mcp/servers/${serverId}/test`, {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                this.showNotification('success', 'Connection test passed', 'MCP server is responding correctly');
            } else {
                this.showNotification('error', 'Connection test failed', result.error || 'Server is not responding');
            }

        } catch (error) {
            this.showNotification('error', 'Connection test failed', error.message);
        }
    }

    async discoverMCPServers() {
        try {
            const response = await fetch('/api/v1/mcp/discover', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.servers && result.servers.length > 0) {
                this.showNotification('success', 'Servers discovered', `Found ${result.servers.length} MCP servers`);
                this.loadMCPServers();
            } else {
                this.showNotification('info', 'No servers found', 'No new MCP servers discovered on the network');
            }

        } catch (error) {
            this.showNotification('error', 'Discovery failed', error.message);
        }
    }

    openMCPServersModal() {
        const modal = document.getElementById('mcp-servers-modal');
        if (modal) {
            modal.classList.add('active');
            this.loadMCPServers();
        }
    }

    // Notification System
    initNotifications() {
        this.createNotificationContainer();
    }

    createNotificationContainer() {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: var(--spacing-lg);
            right: var(--spacing-lg);
            z-index: 1100;
            pointer-events: none;
        `;
        document.body.appendChild(container);
    }

    showNotification(type, title, message, duration = 5000) {
        const notification = document.createElement('div');
        const id = this.generateId();
        
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    <i class="${this.getNotificationIcon(type)}"></i>
                </div>
                <div class="notification-body">
                    <h4>${title}</h4>
                    <p>${message}</p>
                </div>
            </div>
        `;

        const container = document.getElementById('notification-container');
        container.appendChild(notification);

        // Trigger animation
        setTimeout(() => {
            notification.classList.add('show');
            notification.style.pointerEvents = 'all';
        }, 100);

        // Auto-remove
        setTimeout(() => {
            this.removeNotification(notification);
        }, duration);

        // Store reference
        this.notifications.push({ id, element: notification });

        return id;
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    removeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }

    // Utility Methods
    generateId() {
        return Math.random().toString(36).substring(2) + Date.now().toString(36);
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('active');
        }
    }

    getGoogleClientId() {
        // This should be configured in your application settings
        return '1234567890-example.apps.googleusercontent.com';
    }

    getDropboxClientId() {
        // This should be configured in your application settings
        return 'your_dropbox_app_key';
    }

    getMicrosoftClientId() {
        // This should be configured in your application settings
        return 'your_microsoft_app_id';
    }

    async waitForAuthCallback(popup, provider) {
        return new Promise((resolve, reject) => {
            const checkClosed = setInterval(() => {
                if (popup.closed) {
                    clearInterval(checkClosed);
                    reject(new Error('Authentication cancelled'));
                }
            }, 1000);

            window.addEventListener('message', (event) => {
                if (event.data.type === `${provider}_auth_success`) {
                    clearInterval(checkClosed);
                    popup.close();
                    resolve(event.data.accessToken);
                } else if (event.data.type === `${provider}_auth_error`) {
                    clearInterval(checkClosed);
                    popup.close();
                    reject(new Error(event.data.error));
                }
            });
        });
    }
}

// Global functions for onclick handlers
function closeModal(modalId) {
    if (window.enhancedFeatures) {
        window.enhancedFeatures.closeModal(modalId);
    }
}

// Initialize enhanced features when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (window.openSyntheticsApp) {
        window.enhancedFeatures = new EnhancedFeatures(window.openSyntheticsApp);
    }
}); 