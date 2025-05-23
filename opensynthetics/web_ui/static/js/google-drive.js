/**
 * OpenSynthetics Google Drive Integration
 * Secure OAuth2 authentication and file management
 */

class GoogleDriveIntegration {
    constructor(enhancedFeatures) {
        this.enhancedFeatures = enhancedFeatures;
        this.gapi = null;
        this.isSignedIn = false;
        this.currentUser = null;
        this.authInstance = null;
        this.driveApi = null;
        
        // Configuration
        this.config = {
            clientId: '',
            apiKey: '',
            scope: 'https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/drive.metadata.readonly',
            discoveryDocs: [
                'https://www.googleapis.com/discovery/v1/apis/drive/v3/rest'
            ]
        };
        
        this.init();
    }

    async init() {
        try {
            await this.loadConfiguration();
            await this.initializeGoogleAPI();
            this.createGoogleDrivePage();
            this.setupEventListeners();
        } catch (error) {
            console.error('Failed to initialize Google Drive integration:', error);
        }
    }

    async loadConfiguration() {
        try {
            const response = await fetch('/api/v1/integrations/google-drive/config');
            const config = await response.json();
            
            if (config.client_id && config.api_key) {
                this.config.clientId = config.client_id;
                this.config.apiKey = config.api_key;
            } else {
                throw new Error('Google Drive configuration not found');
            }
        } catch (error) {
            console.warn('Google Drive configuration not available:', error);
            this.showConfigurationModal();
        }
    }

    async initializeGoogleAPI() {
        if (!this.config.clientId || !this.config.apiKey) {
            throw new Error('Google Drive not configured');
        }

        return new Promise((resolve, reject) => {
            gapi.load('auth2:client:picker', async () => {
                try {
                    await gapi.client.init({
                        apiKey: this.config.apiKey,
                        clientId: this.config.clientId,
                        discoveryDocs: this.config.discoveryDocs,
                        scope: this.config.scope
                    });

                    this.authInstance = gapi.auth2.getAuthInstance();
                    this.isSignedIn = this.authInstance.isSignedIn.get();
                    
                    if (this.isSignedIn) {
                        this.currentUser = this.authInstance.currentUser.get();
                        this.updateAuthStatus();
                    }

                    // Listen for sign-in state changes
                    this.authInstance.isSignedIn.listen(this.updateAuthStatus.bind(this));

                    resolve();
                } catch (error) {
                    reject(error);
                }
            });
        });
    }

    createGoogleDrivePage() {
        const pageContent = `
            <div id="google-drive-page" class="page">
                <div class="page-header">
                    <h1 class="page-title">Google Drive Integration</h1>
                    <p class="page-subtitle">Securely connect and manage your Google Drive files</p>
                </div>

                <!-- Authentication Section -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Authentication</h3>
                        <div id="auth-status">
                            <span class="status-indicator" id="drive-status-indicator"></span>
                            <span id="drive-status-text">Checking connection...</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="signed-out-content">
                            <p>Connect your Google Drive account to upload, download, and manage files directly from OpenSynthetics.</p>
                            <div style="margin-top: var(--spacing-lg);">
                                <button class="btn btn-primary" onclick="googleDrive.signIn()">
                                    <i class="fab fa-google"></i>
                                    Sign in with Google
                                </button>
                                <button class="btn btn-secondary" onclick="googleDrive.showConfigurationModal()">
                                    <i class="fas fa-cog"></i>
                                    Configure API Keys
                                </button>
                            </div>
                        </div>
                        
                        <div id="signed-in-content" class="hidden">
                            <div class="d-flex align-center gap-md">
                                <img id="user-avatar" src="" alt="User Avatar" style="width: 40px; height: 40px; border-radius: 50%;">
                                <div>
                                    <div id="user-name" style="font-weight: 600;"></div>
                                    <div id="user-email" style="color: var(--text-secondary); font-size: 0.9rem;"></div>
                                </div>
                            </div>
                            <div style="margin-top: var(--spacing-lg);">
                                <button class="btn btn-secondary" onclick="googleDrive.signOut()">
                                    <i class="fas fa-sign-out-alt"></i>
                                    Sign Out
                                </button>
                                <button class="btn btn-primary" onclick="googleDrive.refreshFiles()">
                                    <i class="fas fa-sync"></i>
                                    Refresh Files
                                </button>
                                <button class="btn btn-success" onclick="googleDrive.uploadFile()">
                                    <i class="fas fa-upload"></i>
                                    Upload to Drive
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- File Browser Section -->
                <div class="card" id="file-browser-card" style="display: none;">
                    <div class="card-header">
                        <h3 class="card-title">Files & Folders</h3>
                        <div class="d-flex gap-sm">
                            <select id="folder-selector" class="form-select" style="width: auto;" onchange="googleDrive.changeFolder()">
                                <option value="root">My Drive (Root)</option>
                            </select>
                            <button class="btn btn-sm btn-secondary" onclick="googleDrive.createFolder()">
                                <i class="fas fa-folder-plus"></i>
                                New Folder
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="files-loading" class="text-center" style="padding: var(--spacing-xl);">
                            <i class="fas fa-spinner fa-spin fa-2x"></i>
                            <p style="margin-top: var(--spacing-md);">Loading files...</p>
                        </div>
                        
                        <div id="files-grid" class="hidden" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--spacing-md);"></div>
                        
                        <div id="files-empty" class="hidden text-center" style="padding: var(--spacing-xl);">
                            <i class="fas fa-folder-open fa-3x" style="color: var(--text-tertiary);"></i>
                            <h4 style="margin-top: var(--spacing-md);">No files found</h4>
                            <p style="color: var(--text-secondary);">This folder is empty or you don't have access to view its contents.</p>
                        </div>
                    </div>
                </div>

                <!-- File Operations Section -->
                <div class="card" id="file-operations-card" style="display: none;">
                    <div class="card-header">
                        <h3 class="card-title">File Operations</h3>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-2">
                            <div>
                                <h4>Upload Files</h4>
                                <p style="color: var(--text-secondary);">Upload files from your computer to Google Drive</p>
                                <button class="btn btn-primary" onclick="googleDrive.uploadFile()">
                                    <i class="fas fa-upload"></i>
                                    Upload Files
                                </button>
                            </div>
                            <div>
                                <h4>Sync with OpenSynthetics</h4>
                                <p style="color: var(--text-secondary);">Import data files directly into your workspace</p>
                                <button class="btn btn-secondary" onclick="googleDrive.syncToWorkspace()">
                                    <i class="fas fa-sync"></i>
                                    Sync Selected Files
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Storage Info Section -->
                <div class="card" id="storage-info-card" style="display: none;">
                    <div class="card-header">
                        <h3 class="card-title">Storage Information</h3>
                    </div>
                    <div class="card-body">
                        <div id="storage-info">
                            <div class="d-flex justify-between mb-lg">
                                <span>Used Storage:</span>
                                <span id="used-storage">-</span>
                            </div>
                            <div class="d-flex justify-between mb-lg">
                                <span>Total Storage:</span>
                                <span id="total-storage">-</span>
                            </div>
                            <div class="progress-bar" style="margin-top: var(--spacing-md);">
                                <div class="progress-fill" id="storage-progress" style="width: 0%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Insert the page content into the main content area
        const dynamicContent = document.getElementById('dynamic-content');
        if (dynamicContent) {
            dynamicContent.insertAdjacentHTML('beforeend', pageContent);
        }
    }

    setupEventListeners() {
        // Add global keyboard shortcuts for Google Drive
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'G') {
                e.preventDefault();
                this.enhancedFeatures.app.showPage('google-drive');
            }
        });
    }

    updateAuthStatus() {
        this.isSignedIn = this.authInstance.isSignedIn.get();
        const statusIndicator = document.getElementById('drive-status-indicator');
        const statusText = document.getElementById('drive-status-text');
        const signedOutContent = document.getElementById('signed-out-content');
        const signedInContent = document.getElementById('signed-in-content');
        
        if (this.isSignedIn) {
            this.currentUser = this.authInstance.currentUser.get();
            const profile = this.currentUser.getBasicProfile();
            
            // Update status
            statusIndicator.className = 'status-indicator connected';
            statusText.textContent = 'Connected to Google Drive';
            
            // Update user info
            document.getElementById('user-avatar').src = profile.getImageUrl();
            document.getElementById('user-name').textContent = profile.getName();
            document.getElementById('user-email').textContent = profile.getEmail();
            
            // Show/hide content
            signedOutContent.classList.add('hidden');
            signedInContent.classList.remove('hidden');
            
            // Show additional sections
            document.getElementById('file-browser-card').style.display = 'block';
            document.getElementById('file-operations-card').style.display = 'block';
            document.getElementById('storage-info-card').style.display = 'block';
            
            // Load initial data
            this.loadFiles();
            this.loadStorageInfo();
            
        } else {
            // Update status
            statusIndicator.className = 'status-indicator error';
            statusText.textContent = 'Not connected';
            
            // Show/hide content
            signedOutContent.classList.remove('hidden');
            signedInContent.classList.add('hidden');
            
            // Hide additional sections
            document.getElementById('file-browser-card').style.display = 'none';
            document.getElementById('file-operations-card').style.display = 'none';
            document.getElementById('storage-info-card').style.display = 'none';
        }
    }

    async signIn() {
        try {
            if (!this.authInstance) {
                await this.initializeGoogleAPI();
            }
            
            await this.authInstance.signIn();
            this.enhancedFeatures.showNotification('success', 'Signed In', 'Successfully connected to Google Drive');
        } catch (error) {
            console.error('Sign in failed:', error);
            this.enhancedFeatures.showNotification('error', 'Sign In Failed', error.message);
        }
    }

    async signOut() {
        try {
            await this.authInstance.signOut();
            this.enhancedFeatures.showNotification('info', 'Signed Out', 'Disconnected from Google Drive');
        } catch (error) {
            console.error('Sign out failed:', error);
            this.enhancedFeatures.showNotification('error', 'Sign Out Failed', error.message);
        }
    }

    async loadFiles(folderId = 'root') {
        try {
            const filesLoading = document.getElementById('files-loading');
            const filesGrid = document.getElementById('files-grid');
            const filesEmpty = document.getElementById('files-empty');
            
            filesLoading.classList.remove('hidden');
            filesGrid.classList.add('hidden');
            filesEmpty.classList.add('hidden');

            const response = await gapi.client.drive.files.list({
                q: `'${folderId}' in parents and trashed=false`,
                pageSize: 50,
                fields: 'nextPageToken, files(id, name, mimeType, size, modifiedTime, thumbnailLink, webViewLink, parents)',
                orderBy: 'folder,name'
            });

            const files = response.result.files;
            
            filesLoading.classList.add('hidden');
            
            if (files && files.length > 0) {
                this.renderFiles(files);
                filesGrid.classList.remove('hidden');
            } else {
                filesEmpty.classList.remove('hidden');
            }
            
        } catch (error) {
            console.error('Failed to load files:', error);
            this.enhancedFeatures.showNotification('error', 'Failed to load files', error.message);
        }
    }

    renderFiles(files) {
        const filesGrid = document.getElementById('files-grid');
        
        filesGrid.innerHTML = files.map(file => {
            const isFolder = file.mimeType === 'application/vnd.google-apps.folder';
            const icon = this.getFileIcon(file.mimeType);
            const size = file.size ? this.formatFileSize(parseInt(file.size)) : '-';
            const modifiedDate = new Date(file.modifiedTime).toLocaleDateString();
            
            return `
                <div class="file-item-card" data-file-id="${file.id}" style="
                    border: 1px solid var(--border-color);
                    border-radius: var(--border-radius);
                    padding: var(--spacing-md);
                    cursor: pointer;
                    transition: var(--transition);
                    background-color: var(--surface-color);
                " onmouseover="this.style.borderColor='var(--primary-color)'" 
                   onmouseout="this.style.borderColor='var(--border-color)'"
                   onclick="googleDrive.${isFolder ? 'openFolder' : 'selectFile'}('${file.id}')">
                    
                    <div class="d-flex align-center gap-md mb-lg">
                        <div class="file-icon" style="font-size: 1.5rem; color: var(--primary-color);">
                            <i class="${icon}"></i>
                        </div>
                        <div style="flex: 1; min-width: 0;">
                            <div class="file-name text-truncate" style="font-weight: 500;">${file.name}</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                                ${isFolder ? 'Folder' : size} • ${modifiedDate}
                            </div>
                        </div>
                    </div>
                    
                    <div class="file-actions d-flex gap-sm">
                        ${!isFolder ? `
                            <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); googleDrive.previewFile('${file.id}')">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); googleDrive.downloadFile('${file.id}')">
                                <i class="fas fa-download"></i>
                            </button>
                        ` : `
                            <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); googleDrive.openFolder('${file.id}')">
                                <i class="fas fa-folder-open"></i>
                            </button>
                        `}
                        <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); googleDrive.shareFile('${file.id}')">
                            <i class="fas fa-share"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    }

    getFileIcon(mimeType) {
        const iconMap = {
            'application/vnd.google-apps.folder': 'fas fa-folder',
            'application/vnd.google-apps.document': 'fas fa-file-word',
            'application/vnd.google-apps.spreadsheet': 'fas fa-file-excel',
            'application/vnd.google-apps.presentation': 'fas fa-file-powerpoint',
            'application/pdf': 'fas fa-file-pdf',
            'text/csv': 'fas fa-file-csv',
            'application/json': 'fas fa-file-code',
            'text/plain': 'fas fa-file-alt',
            'image/jpeg': 'fas fa-file-image',
            'image/png': 'fas fa-file-image',
            'image/gif': 'fas fa-file-image',
            'video/mp4': 'fas fa-file-video',
            'audio/mpeg': 'fas fa-file-audio'
        };
        
        if (mimeType.startsWith('image/')) return 'fas fa-file-image';
        if (mimeType.startsWith('video/')) return 'fas fa-file-video';
        if (mimeType.startsWith('audio/')) return 'fas fa-file-audio';
        if (mimeType.startsWith('text/')) return 'fas fa-file-alt';
        
        return iconMap[mimeType] || 'fas fa-file';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async loadStorageInfo() {
        try {
            const response = await gapi.client.drive.about.get({
                fields: 'storageQuota'
            });
            
            const quota = response.result.storageQuota;
            const used = parseInt(quota.usage || 0);
            const total = parseInt(quota.limit || 0);
            
            document.getElementById('used-storage').textContent = this.formatFileSize(used);
            document.getElementById('total-storage').textContent = total > 0 ? this.formatFileSize(total) : 'Unlimited';
            
            if (total > 0) {
                const percentage = (used / total) * 100;
                document.getElementById('storage-progress').style.width = `${percentage}%`;
            }
            
        } catch (error) {
            console.error('Failed to load storage info:', error);
        }
    }

    async uploadFile() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        
        input.onchange = async (e) => {
            const files = Array.from(e.target.files);
            
            for (const file of files) {
                try {
                    await this.uploadSingleFile(file);
                } catch (error) {
                    console.error(`Failed to upload ${file.name}:`, error);
                    this.enhancedFeatures.showNotification('error', 'Upload Failed', `Failed to upload ${file.name}`);
                }
            }
            
            this.refreshFiles();
        };
        
        input.click();
    }

    async uploadSingleFile(file) {
        const metadata = {
            name: file.name,
            parents: [this.getCurrentFolderId()]
        };

        const form = new FormData();
        form.append('metadata', new Blob([JSON.stringify(metadata)], {type: 'application/json'}));
        form.append('file', file);

        const response = await fetch('https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart', {
            method: 'POST',
            headers: new Headers({
                'Authorization': `Bearer ${this.currentUser.getAuthResponse().access_token}`
            }),
            body: form
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        this.enhancedFeatures.showNotification('success', 'Upload Complete', `Successfully uploaded ${file.name}`);
        return await response.json();
    }

    getCurrentFolderId() {
        const folderSelector = document.getElementById('folder-selector');
        return folderSelector ? folderSelector.value : 'root';
    }

    async downloadFile(fileId) {
        try {
            const response = await gapi.client.drive.files.get({
                fileId: fileId,
                alt: 'media'
            });

            // Create download link
            const blob = new Blob([response.body], { type: response.headers['Content-Type'] });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.result.name || 'download';
            a.click();
            window.URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Download failed:', error);
            this.enhancedFeatures.showNotification('error', 'Download Failed', error.message);
        }
    }

    async createFolder() {
        const name = prompt('Enter folder name:');
        if (!name) return;

        try {
            const metadata = {
                name: name,
                mimeType: 'application/vnd.google-apps.folder',
                parents: [this.getCurrentFolderId()]
            };

            await gapi.client.drive.files.create({
                resource: metadata
            });

            this.enhancedFeatures.showNotification('success', 'Folder Created', `Successfully created folder: ${name}`);
            this.refreshFiles();

        } catch (error) {
            console.error('Failed to create folder:', error);
            this.enhancedFeatures.showNotification('error', 'Folder Creation Failed', error.message);
        }
    }

    refreshFiles() {
        this.loadFiles(this.getCurrentFolderId());
    }

    showConfigurationModal() {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.id = 'google-drive-config-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">Configure Google Drive API</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="form-group">
                        <label class="form-label required">Google Client ID</label>
                        <input type="text" class="form-input" name="client_id" placeholder="your-client-id.apps.googleusercontent.com" required>
                        <div class="form-help">Get this from Google Cloud Console → APIs & Services → Credentials</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label required">Google API Key</label>
                        <input type="password" class="form-input" name="api_key" required>
                        <div class="form-help">Your Google Drive API key from Google Cloud Console</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Instructions</label>
                        <div style="background: var(--surface-secondary); padding: var(--spacing-md); border-radius: var(--border-radius); font-size: 0.9rem;">
                            <ol style="margin: 0; padding-left: var(--spacing-lg);">
                                <li>Go to <a href="https://console.cloud.google.com/" target="_blank">Google Cloud Console</a></li>
                                <li>Create a new project or select existing one</li>
                                <li>Enable the Google Drive API</li>
                                <li>Go to "APIs & Services" → "Credentials"</li>
                                <li>Create an OAuth 2.0 Client ID (Web application)</li>
                                <li>Add your domain to authorized JavaScript origins</li>
                                <li>Create an API Key and restrict it to Google Drive API</li>
                            </ol>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                    <button class="btn btn-primary" onclick="googleDrive.saveConfiguration()">Save Configuration</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async saveConfiguration() {
        const modal = document.getElementById('google-drive-config-modal');
        const clientId = modal.querySelector('input[name="client_id"]').value;
        const apiKey = modal.querySelector('input[name="api_key"]').value;

        if (!clientId || !apiKey) {
            this.enhancedFeatures.showNotification('error', 'Invalid Configuration', 'Please provide both Client ID and API Key');
            return;
        }

        try {
            const response = await fetch('/api/v1/integrations/google-drive/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    client_id: clientId,
                    api_key: apiKey
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to save configuration: ${response.statusText}`);
            }

            this.config.clientId = clientId;
            this.config.apiKey = apiKey;

            modal.remove();
            this.enhancedFeatures.showNotification('success', 'Configuration Saved', 'Google Drive API configured successfully');
            
            // Reinitialize with new config
            await this.initializeGoogleAPI();

        } catch (error) {
            console.error('Failed to save configuration:', error);
            this.enhancedFeatures.showNotification('error', 'Configuration Failed', error.message);
        }
    }

    // Additional utility methods
    async previewFile(fileId) {
        try {
            const response = await gapi.client.drive.files.get({
                fileId: fileId,
                fields: 'name, mimeType, webViewLink, size'
            });

            const file = response.result;
            
            if (file.webViewLink) {
                window.open(file.webViewLink, '_blank');
            } else {
                this.enhancedFeatures.showNotification('info', 'Preview Not Available', 'This file cannot be previewed in the browser');
            }

        } catch (error) {
            console.error('Preview failed:', error);
            this.enhancedFeatures.showNotification('error', 'Preview Failed', error.message);
        }
    }

    async shareFile(fileId) {
        try {
            const response = await gapi.client.drive.files.get({
                fileId: fileId,
                fields: 'name, webViewLink'
            });

            const file = response.result;
            
            if (navigator.share) {
                await navigator.share({
                    title: file.name,
                    url: file.webViewLink
                });
            } else {
                // Fallback: copy to clipboard
                await navigator.clipboard.writeText(file.webViewLink);
                this.enhancedFeatures.showNotification('success', 'Link Copied', 'File link copied to clipboard');
            }

        } catch (error) {
            console.error('Share failed:', error);
            this.enhancedFeatures.showNotification('error', 'Share Failed', error.message);
        }
    }

    async openFolder(folderId) {
        document.getElementById('folder-selector').value = folderId;
        await this.loadFiles(folderId);
    }

    async syncToWorkspace() {
        // This would integrate with the main workspace system
        this.enhancedFeatures.showNotification('info', 'Sync Feature', 'File sync functionality will be implemented in the workspace integration');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (window.enhancedFeatures) {
        window.googleDrive = new GoogleDriveIntegration(window.enhancedFeatures);
    }
}); 