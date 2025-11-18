// Video Chat Application
class VideoChat {
    constructor() {
        this.currentVideoId = null;
        this.videoStatus = null;
        this.statusCheckInterval = null;
        this.currentTab = 'main';

        this.initElements();
        this.attachEventListeners();
        this.loadVideoList();
        this.loadCosts();

        // Refresh costs every 5 seconds when on costs tab
        setInterval(() => {
            if (this.currentTab === 'costs') {
                this.loadCosts();
            }
        }, 5000);
    }

    initElements() {
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.uploadProgress = document.getElementById('upload-progress');
        this.progressBar = document.getElementById('progress-bar');
        this.progressStatus = document.getElementById('progress-status');

        // Library elements
        this.videoLibrary = document.getElementById('video-library');
        this.videoList = document.getElementById('video-list');
        this.newUploadBtn = document.getElementById('new-upload-btn');

        // Video elements
        this.videoPlayer = document.getElementById('video-player');
        this.videoInfo = document.getElementById('video-info');
        this.timelineContainer = document.getElementById('timeline-container');
        this.timelineStrip = document.getElementById('timeline-strip');
        this.highlightsTimeline = document.getElementById('highlights-timeline');
        this.keyframesGrid = document.getElementById('keyframes-grid');

        // Chat elements
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.quickActions = document.getElementById('quick-actions');

        // Info elements
        this.infoFilename = document.getElementById('info-filename');
        this.infoDuration = document.getElementById('info-duration');
        this.infoStatus = document.getElementById('info-status');
    }

    attachEventListeners() {
        // Tab switching
        document.querySelectorAll('.header-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Upload area
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragging');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragging');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragging');
            if (e.dataTransfer.files.length > 0) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        // Chat
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Quick actions
        this.quickActions.addEventListener('click', (e) => {
            if (e.target.classList.contains('quick-action-btn')) {
                const query = e.target.dataset.query;
                this.chatInput.value = query;
                this.sendMessage();
            }
        });

        // New upload button
        this.newUploadBtn.addEventListener('click', () => {
            this.showUploadArea();
        });
    }

    async loadVideoList() {
        try {
            const response = await fetch('/api/videos');
            const videos = await response.json();

            // Show library if there are videos
            if (videos.length > 0) {
                this.videoLibrary.classList.add('active');
                this.renderVideoLibrary(videos);

                // Load the most recent ready video
                const latestVideo = videos.find(v => v.status === 'ready') || videos[0];
                if (latestVideo) {
                    await this.loadVideo(latestVideo.id);
                    this.uploadArea.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('Error loading video list:', error);
        }
    }

    renderVideoLibrary(videos) {
        // Deduplicate videos by ID (keep the latest entry)
        const uniqueVideos = {};
        videos.forEach(video => {
            uniqueVideos[video.id] = video;
        });
        const deduplicatedVideos = Object.values(uniqueVideos);

        this.videoList.innerHTML = deduplicatedVideos.map(video => {
            const isActive = video.id === this.currentVideoId;
            const statusClass = video.status === 'ready' ? 'ready' :
                               video.status === 'error' ? 'error' : 'processing';

            // Show reprocess button for completed or failed videos
            const showReprocess = video.status === 'ready' || video.status === 'error';

            // Show delete button for failed videos
            const showDelete = video.status === 'error' || video.status === 'ready';

            return `
                <div class="video-item ${isActive ? 'active' : ''}" data-video-id="${video.id}">
                    <div class="video-item-info">
                        <div class="video-item-name">${video.filename}</div>
                        <div class="video-item-meta">
                            ${video.duration_seconds ? this.formatTime(video.duration_seconds) : 'Processing...'}
                        </div>
                    </div>
                    <div class="video-item-actions">
                        <span class="video-item-status status-badge ${statusClass}">
                            ${video.status}
                        </span>
                        ${showReprocess ? `<button class="video-action-btn reprocess-btn" data-video-id="${video.id}" title="Reprocess video">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
                            </svg>
                        </button>` : ''}
                        ${showDelete ? `<button class="video-action-btn delete-btn" data-video-id="${video.id}" title="Delete video">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers for video items
        this.videoList.querySelectorAll('.video-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't load video if clicking on action buttons
                if (e.target.closest('.video-action-btn')) return;

                const videoId = parseInt(item.dataset.videoId);
                this.loadVideo(videoId);
                this.uploadArea.style.display = 'none';
            });
        });

        // Add handlers for reprocess buttons
        this.videoList.querySelectorAll('.reprocess-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const videoId = parseInt(btn.dataset.videoId);
                this.reprocessVideo(videoId);
            });
        });

        // Add handlers for delete buttons
        this.videoList.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const videoId = parseInt(btn.dataset.videoId);
                this.deleteVideo(videoId);
            });
        });
    }

    showUploadArea() {
        this.uploadArea.style.display = 'block';
        this.videoPlayer.classList.remove('active');
        this.videoInfo.classList.remove('active');
        this.timelineContainer.classList.remove('active');
        this.chatInput.disabled = true;
        this.sendBtn.disabled = true;
        this.quickActions.style.display = 'none';
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFileUpload(file);
        }
    }

    async handleFileUpload(file) {
        // Validate file size (2GB)
        const maxSize = 2 * 1024 * 1024 * 1024;
        if (file.size > maxSize) {
            this.addSystemMessage('File too large! Maximum size is 2GB.');
            return;
        }

        this.addSystemMessage(`Uploading ${file.name}...`);

        // Show progress
        this.uploadArea.classList.add('uploading');
        this.uploadProgress.classList.add('active');

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Use XMLHttpRequest for progress tracking
            await new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();

                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = Math.round((e.loaded / e.total) * 100);
                        this.updateProgress(percentComplete, 'Uploading...');
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.detail || 'Upload failed'));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error during upload'));
                });

                xhr.open('POST', '/api/upload');
                xhr.send(formData);
            }).then(result => {
                this.updateProgress(100, 'Upload complete! Processing...');
                this.addSystemMessage(
                    `✅ ${result.filename} uploaded successfully! Processing video...`
                );

                this.currentVideoId = result.video_id;
                this.uploadArea.style.display = 'none';
                this.startStatusPolling();

                // Refresh video library
                this.loadVideoList();

                // Reset upload area after a delay
                setTimeout(() => {
                    this.uploadArea.classList.remove('uploading');
                    this.uploadProgress.classList.remove('active');
                    this.updateProgress(0, '');
                }, 2000);
            });

        } catch (error) {
            console.error('Upload error:', error);
            this.addSystemMessage(`❌ Upload failed: ${error.message}`);
            this.uploadArea.classList.remove('uploading');
            this.uploadProgress.classList.remove('active');
            this.updateProgress(0, '');
        }
    }

    updateProgress(percent, status) {
        this.progressBar.style.width = `${percent}%`;
        this.progressBar.textContent = `${percent}%`;
        this.progressStatus.textContent = status;
    }

    startStatusPolling() {
        // Check status every 2 seconds
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }

        this.statusCheckInterval = setInterval(async () => {
            await this.checkVideoStatus();
        }, 2000);
    }

    async checkVideoStatus() {
        if (!this.currentVideoId) return;

        try {
            const response = await fetch(`/api/videos/${this.currentVideoId}`);
            const videoData = await response.json();

            this.videoStatus = videoData.status;
            this.updateStatusDisplay(videoData);

            if (videoData.status === 'ready') {
                clearInterval(this.statusCheckInterval);
                await this.loadVideo(this.currentVideoId);
                this.addSystemMessage('✅ Video ready! You can now chat with it.');
                this.loadVideoList(); // Refresh library
            } else if (videoData.status === 'error') {
                clearInterval(this.statusCheckInterval);
                this.addSystemMessage('❌ Video processing failed.');
                this.loadVideoList(); // Refresh library
            }

        } catch (error) {
            console.error('Error checking status:', error);
        }
    }

    updateStatusDisplay(videoData) {
        // Update info panel
        this.videoInfo.classList.add('active');
        this.infoFilename.textContent = videoData.filename;
        this.infoDuration.textContent = videoData.duration_seconds
            ? this.formatTime(videoData.duration_seconds)
            : '-';

        // Update status badge
        const statusBadges = {
            'processing': '<span class="status-badge processing">Processing...</span>',
            'analyzing': '<span class="status-badge processing">Analyzing...</span>',
            'indexing': '<span class="status-badge processing">Indexing...</span>',
            'ready': '<span class="status-badge ready">Ready</span>',
            'error': '<span class="status-badge error">Error</span>'
        };

        this.infoStatus.innerHTML = statusBadges[videoData.status] || videoData.status;
    }

    async loadVideo(videoId) {
        try {
            const response = await fetch(`/api/videos/${videoId}`);
            const videoData = await response.json();

            this.currentVideoId = videoId;
            this.videoStatus = videoData.status;

            // Update video player
            this.videoPlayer.src = videoData.video_url;
            this.videoPlayer.classList.add('active');

            // Update info
            this.updateStatusDisplay(videoData);

            // Load timeline and keyframes
            if (videoData.status === 'ready') {
                await this.loadTimeline(videoId);
                await this.loadKeyframes(videoId);

                // Enable chat
                this.chatInput.disabled = false;
                this.sendBtn.disabled = false;
                this.quickActions.style.display = 'flex';

                this.addSystemMessage('Video loaded! Ask me anything about it.');
            }

        } catch (error) {
            console.error('Error loading video:', error);
            this.addSystemMessage('Failed to load video.');
        }
    }

    async loadTimeline(videoId) {
        try {
            // Load ALL frames for interactive timeline
            const response = await fetch(`/api/videos/${videoId}/frames`);
            const frames = await response.json();

            // Clear timeline
            this.timelineStrip.innerHTML = '';

            // Sample frames for timeline (show every 10th frame or max 30 frames)
            const sampleRate = Math.max(1, Math.floor(frames.length / 30));
            const timelineFrames = frames.filter((_, index) => index % sampleRate === 0);

            timelineFrames.forEach(frame => {
                const frameEl = document.createElement('div');
                frameEl.className = 'timeline-frame';

                // Ensure click handler is properly attached
                frameEl.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log(`Seeking to ${frame.timestamp_seconds}s`);
                    this.seekToTime(frame.timestamp_seconds);
                });

                // Build tooltip content
                const tooltipText = frame.description || 'No description available';
                const excitementBadge = frame.excitement_score
                    ? `<div style="margin-top: 4px;"><span style="background: ${frame.excitement_score >= 7 ? '#f39c12' : '#95a5a6'}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.75em;">⚡ ${frame.excitement_score}/10</span></div>`
                    : '';

                frameEl.innerHTML = `
                    <img src="${frame.thumbnail_url}" alt="Frame at ${this.formatTime(frame.timestamp_seconds)}" style="pointer-events: none;">
                    <div class="timestamp-label">${this.formatTime(frame.timestamp_seconds)}</div>
                    <div class="frame-tooltip">
                        <strong>${this.formatTime(frame.timestamp_seconds)}</strong>
                        <div style="margin-top: 6px;">${tooltipText}</div>
                        ${excitementBadge}
                    </div>
                `;

                this.timelineStrip.appendChild(frameEl);
            });

            // Load highlights
            await this.loadHighlights(videoId);

            this.timelineContainer.classList.add('active');
        } catch (error) {
            console.error('Error loading timeline:', error);
        }
    }

    async loadHighlights(videoId) {
        try {
            // Fetch frames with high excitement scores
            const response = await fetch(`/api/videos/${videoId}/frames`);
            const frames = await response.json();

            // Filter for exciting frames (score >= 7)
            const highlights = frames
                .filter(f => f.excitement_score && f.excitement_score >= 7)
                .sort((a, b) => b.excitement_score - a.excitement_score)
                .slice(0, 10); // Top 10 highlights

            this.highlightsTimeline.innerHTML = '';

            if (highlights.length === 0) {
                this.highlightsTimeline.innerHTML = `
                    <div style="text-align: center; padding: 20px; color: #999;">
                        No highlights detected yet. Try analyzing the video first!
                    </div>
                `;
                return;
            }

            highlights.forEach(frame => {
                const highlightEl = document.createElement('div');
                highlightEl.className = 'highlight-item';

                // Proper click handler
                highlightEl.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log(`Jumping to highlight at ${frame.timestamp_seconds}s`);
                    this.seekToTime(frame.timestamp_seconds);
                });

                highlightEl.innerHTML = `
                    <img src="${frame.thumbnail_url}" alt="Highlight" class="highlight-thumbnail" style="pointer-events: none;">
                    <div class="highlight-info">
                        <div class="highlight-time">
                            ${this.formatTime(frame.timestamp_seconds)}
                            <span class="highlight-score">⚡ ${frame.excitement_score}/10</span>
                        </div>
                        <div class="highlight-description">${frame.description || 'Exciting moment detected!'}</div>
                    </div>
                `;

                this.highlightsTimeline.appendChild(highlightEl);
            });

        } catch (error) {
            console.error('Error loading highlights:', error);
        }
    }

    async loadKeyframes(videoId) {
        try {
            const response = await fetch(`/api/videos/${videoId}/frames?keyframes_only=true`);
            const frames = await response.json();

            this.keyframesGrid.innerHTML = '';

            frames.forEach((frame, index) => {
                const card = document.createElement('div');
                card.className = 'keyframe-card';

                // Proper click handler
                card.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log(`Jumping to keyframe at ${frame.timestamp_seconds}s`);
                    this.seekToTime(frame.timestamp_seconds);
                });

                card.innerHTML = `
                    <img src="${frame.thumbnail_url}" alt="Frame at ${this.formatTime(frame.timestamp_seconds)}" style="pointer-events: none;">
                    <div class="keyframe-time">${this.formatTime(frame.timestamp_seconds)}</div>
                `;

                this.keyframesGrid.appendChild(card);
            });

        } catch (error) {
            console.error('Error loading keyframes:', error);
        }
    }

    seekToTime(seconds) {
        if (!this.videoPlayer || !this.videoPlayer.src) {
            console.error('Video player not ready');
            return;
        }

        console.log(`Seeking video to ${seconds} seconds`);
        this.videoPlayer.currentTime = seconds;

        // Scroll video into view if needed
        this.videoPlayer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Play the video
        this.videoPlayer.play().catch(err => {
            console.warn('Autoplay prevented, video paused at timestamp');
        });
    }

    async sendMessage() {
        const query = this.chatInput.value.trim();
        if (!query || !this.currentVideoId) return;

        // Add user message
        this.addUserMessage(query);
        this.chatInput.value = '';

        // Disable input while processing
        this.chatInput.disabled = true;
        this.sendBtn.disabled = true;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_id: this.currentVideoId,
                    query: query
                })
            });

            if (!response.ok) {
                throw new Error('Chat request failed');
            }

            const result = await response.json();
            this.addAssistantMessage(result.answer, result.relevant_frames, result.timestamp);

            // Check if this was a highlights/exciting moments query
            const isHighlightsQuery = query.toLowerCase().match(/(highlight|exciting|best|amazing|interesting|action)/);
            if (isHighlightsQuery && result.relevant_frames && result.relevant_frames.length > 0) {
                // Update highlights timeline with these frames
                await this.updateHighlightsFromChat(result.relevant_frames);
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.addSystemMessage('Sorry, I had trouble processing that question.');
        } finally {
            this.chatInput.disabled = false;
            this.sendBtn.disabled = false;
            this.chatInput.focus();
        }
    }

    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addAssistantMessage(text, frames = [], timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        // Add text
        const textSpan = document.createElement('div');
        textSpan.textContent = text;
        messageDiv.appendChild(textSpan);

        // Add frame previews if available
        if (frames && frames.length > 0) {
            const previewDiv = document.createElement('div');
            previewDiv.className = 'frame-preview';

            frames.forEach(frame => {
                const img = document.createElement('img');
                img.src = frame.image_url;
                img.alt = `Frame at ${this.formatTime(frame.timestamp)}`;
                img.title = `${this.formatTime(frame.timestamp)} - ${frame.description}`;
                img.onclick = () => this.seekToTime(frame.timestamp);
                previewDiv.appendChild(img);
            });

            messageDiv.appendChild(previewDiv);
        }

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addSystemMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.textContent = text;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    async updateHighlightsFromChat(frames) {
        // Fetch full frame data to get excitement scores
        try {
            const response = await fetch(`/api/videos/${this.currentVideoId}/frames`);
            const allFrames = await response.json();

            // Map chat frames to full frame data
            const highlightFrames = frames.map(chatFrame => {
                return allFrames.find(f => Math.abs(f.timestamp_seconds - chatFrame.timestamp) < 0.5) || chatFrame;
            }).filter(f => f);

            // Clear and populate highlights timeline
            this.highlightsTimeline.innerHTML = '';

            if (highlightFrames.length === 0) {
                return;
            }

            highlightFrames.forEach((frame, index) => {
                const highlightEl = document.createElement('div');
                highlightEl.className = 'highlight-item';
                highlightEl.onclick = () => this.seekToTime(frame.timestamp_seconds || frame.timestamp);

                const timestamp = frame.timestamp_seconds || frame.timestamp;
                const score = frame.excitement_score || 8; // Default score for chat results
                const description = frame.description || 'Exciting moment';

                highlightEl.innerHTML = `
                    <img src="${frame.thumbnail_url || frame.image_url}" alt="Highlight" class="highlight-thumbnail">
                    <div class="highlight-info">
                        <div class="highlight-time">
                            ${this.formatTime(timestamp)}
                            <span class="highlight-score">⚡ ${score}/10</span>
                        </div>
                        <div class="highlight-description">${description}</div>
                    </div>
                `;

                this.highlightsTimeline.appendChild(highlightEl);
            });

            // Scroll highlights into view
            this.highlightsTimeline.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (error) {
            console.error('Error updating highlights:', error);
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    async deleteVideo(videoId) {
        if (!confirm('Are you sure you want to delete this video? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`/api/videos/${videoId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to delete video');
            }

            this.addSystemMessage('Video deleted successfully.');

            // Clear current video if it was the deleted one
            if (this.currentVideoId === videoId) {
                this.currentVideoId = null;
                this.videoPlayer.classList.remove('active');
                this.videoInfo.classList.remove('active');
                this.timelineContainer.classList.remove('active');
                this.chatMessages.innerHTML = '';
            }

            // Refresh video list
            await this.loadVideoList();

        } catch (error) {
            console.error('Error deleting video:', error);
            this.addSystemMessage('Failed to delete video.');
        }
    }

    async reprocessVideo(videoId) {
        if (!confirm('Reprocess this video? This will clear existing analysis and start over.')) {
            return;
        }

        try {
            const response = await fetch(`/api/videos/${videoId}/reprocess`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to reprocess video');
            }

            this.addSystemMessage('Video reprocessing started. This may take a few minutes...');

            // Refresh video list
            await this.loadVideoList();

            // If this is the current video, reload it
            if (this.currentVideoId === videoId) {
                // Wait a moment for status to update
                setTimeout(() => {
                    this.loadVideo(videoId);
                }, 1000);
            }

        } catch (error) {
            console.error('Error reprocessing video:', error);
            this.addSystemMessage('Failed to reprocess video.');
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    switchTab(tabName) {
        this.currentTab = tabName;

        // Update tab buttons
        document.querySelectorAll('.header-tab').forEach(tab => {
            if (tab.dataset.tab === tabName) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Update tab content
        document.querySelectorAll('.tab-content, .costs-tab-content').forEach(content => {
            content.classList.remove('active');
        });

        if (tabName === 'main') {
            document.getElementById('main-tab').classList.add('active');
        } else if (tabName === 'costs') {
            document.getElementById('costs-tab').classList.add('active');
            // Force reload costs with cache busting
            this.loadCosts(true);
        }
    }

    async loadCosts(forceRefresh = false) {
        try {
            // Add cache busting parameter if forcing refresh
            const url = forceRefresh
                ? `/api/costs?_=${Date.now()}`
                : '/api/costs';
            const response = await fetch(url, {
                cache: forceRefresh ? 'no-cache' : 'default'
            });
            const data = await response.json();

            // Update summary cards with more decimal places
            document.getElementById('total-cost').textContent =
                `$${data.summary.total_cost_usd.toFixed(6)}`;
            document.getElementById('last-24h-cost').textContent =
                `$${data.summary.last_24h_usd.toFixed(6)}`;

            // Get video stats
            const videosResponse = await fetch('/api/videos');
            const videos = await videosResponse.json();
            const readyVideos = videos.filter(v => v.status === 'ready').length;

            document.getElementById('videos-processed').textContent = readyVideos;
            document.getElementById('avg-cost').textContent =
                readyVideos > 0
                    ? `$${(data.summary.total_cost_usd / readyVideos).toFixed(6)}`
                    : '$0.000000';

            // Update cost log table
            this.updateCostLog(data.recent_operations);

        } catch (error) {
            console.error('Error loading costs:', error);
        }
    }

    async updateCostLog(operations) {
        const tbody = document.getElementById('cost-log-body');

        if (!operations || operations.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; padding: 40px; color: #999;">
                        No operations yet. Upload a video to see cost tracking.
                    </td>
                </tr>
            `;
            return;
        }

        // Get video statuses
        const videosResponse = await fetch('/api/videos');
        const videos = await videosResponse.json();
        const videoStatusMap = {};
        videos.forEach(v => videoStatusMap[v.id] = v.status);

        tbody.innerHTML = operations.map(op => {
            const date = new Date(op.timestamp);
            const timeStr = date.toLocaleTimeString();
            const videoName = op.original_filename || 'N/A';
            const videoStatus = videoStatusMap[op.video_id] || 'unknown';

            // Status badge
            const statusBadge = videoStatus === 'ready'
                ? '<span style="color: #27ae60; font-size: 0.8em;">✓ Ready</span>'
                : videoStatus === 'processing' || videoStatus === 'analyzing' || videoStatus === 'indexing'
                ? '<span style="color: #f39c12; font-size: 0.8em;">⏳ Processing</span>'
                : videoStatus === 'error'
                ? '<span style="color: #e74c3c; font-size: 0.8em;">✗ Error</span>'
                : '<span style="color: #95a5a6; font-size: 0.8em;">○ ' + videoStatus + '</span>';

            // Build token/usage info
            let usageInfo = op.details || '-';
            const usageParts = [];
            if (op.input_tokens > 0) usageParts.push(`${op.input_tokens.toLocaleString()} in`);
            if (op.output_tokens > 0) usageParts.push(`${op.output_tokens.toLocaleString()} out`);
            if (op.num_images > 0) usageParts.push(`${op.num_images} imgs`);
            if (op.num_embeddings > 0) usageParts.push(`${op.num_embeddings} embeds`);

            if (usageParts.length > 0) {
                usageInfo += `<br><span style="color: #999; font-size: 0.85em;">${usageParts.join(' • ')}</span>`;
            }

            return `
                <tr>
                    <td>${timeStr}</td>
                    <td>${this.formatOperationType(op.operation_type)}</td>
                    <td>
                        <div>${videoName}</div>
                        <div style="margin-top: 2px;">${statusBadge}</div>
                    </td>
                    <td>${op.api_provider}</td>
                    <td style="font-size: 0.9em; color: #666;">${usageInfo}</td>
                    <td class="cost-amount">$${op.estimated_cost_usd.toFixed(6)}</td>
                </tr>
            `;
        }).join('');
    }

    formatOperationType(type) {
        const types = {
            'frame_analysis': '🎨 Frame Analysis',
            'embeddings': '🔤 Embeddings',
            'transcription': '🎤 Transcription',
            'chat_query': '💬 Chat Query',
        };
        return types[type] || type;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.videoChat = new VideoChat();
});
