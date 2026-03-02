import { Page, Browser } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Test Helper for Speech-to-Speech (S2S) Conversation Application
 * Handles UI interactions for S2S conversation flow: profile selection → config → start → audio interaction → stop
 */
export class TestHelper {
    private page: Page;

    constructor(page: Page) {
        this.page = page;
    }

    /**
     * Select a profile from the profile dropdown
     * @param profileName Name of the profile to select (e.g., "Nova Sonic v1 - Legacy", "default", etc.)
     */
    async selectProfile(profileName: string): Promise<void> {
        console.log(`Selecting profile: ${profileName}`);

        // Click the AWS UI Select dropdown button (has aria-haspopup="listbox")
        const profileDropdown = this.page.locator('button[aria-haspopup="listbox"]').first();
        await profileDropdown.waitFor({ state: 'visible', timeout: 5000 });
        await profileDropdown.click();

        console.log('  - Dropdown opened');

        // Get all available options
        const options = this.page.getByRole('option');
        const optionCount = await options.count();
        const availableOptions: string[] = [];

        for (let i = 0; i < optionCount; i++) {
            const optionText = await options.nth(i).textContent();
            if (optionText) {
                availableOptions.push(optionText.trim());
            }
        }
        console.log(`  - Available options: ${availableOptions.join(', ')}`);

        // Determine which option to select
        let targetOption = profileName;

        // If profile is "default", select the one marked as default
        if (profileName.toLowerCase() === 'default') {
            targetOption = availableOptions.find(opt => opt.includes('Default')) || availableOptions[availableOptions.length - 1];
            console.log(`  - "default" mapped to: "${targetOption}"`);
        }

        // Click the matching option - match the beginning of the option text
        // (options may have descriptions appended, so we match the start)
        const profileOption = this.page.getByRole('option').filter({ hasText: new RegExp(`${targetOption.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, '') }).first();
        await profileOption.waitFor({ state: 'visible', timeout: 5000 });

        const selectedText = await profileOption.textContent();
        console.log(`  - Selecting: "${selectedText}"`);

        await profileOption.click();

        console.log(`✅ Profile selected: ${profileName}`);
    }

    /**
     * Open the configuration settings modal/panel
     */
    async openConfigSettings(): Promise<void> {
        console.log('Opening config settings...');

        // Wait for page to be stable before looking for config button
        await this.page.waitForLoadState('networkidle').catch(() => {});
        await this.page.waitForTimeout(500);

        // Look for the settings button wrapped in div with class "setting"
        const configButton = this.page.locator('.setting button[type="submit"]');
        await configButton.waitFor({ state: 'visible', timeout: 5000 });
        console.log('  - Settings button found and visible');

        await configButton.click();
        console.log('  - Settings button clicked');

        // Wait for settings modal/panel to appear or settings to open
        try {
            await this.page.waitForSelector('[role="dialog"], .settings-panel, .settings-content', { timeout: 5000 });
            console.log('  - Settings panel appeared');
        } catch {
            console.warn('⚠️ Settings dialog did not appear, checking if settings opened inline...');
        }

        console.log('✅ Config settings opened');
    }

    /**
     * Set the Voice ID in the configuration
     * @param voiceId Voice ID to set
     */
    async setVoiceId(voiceId: string): Promise<void> {
        console.log(`Setting voice ID: ${voiceId}`);

        // Voice selection uses a button trigger with aria-haspopup="dialog"
        // Find the button that triggers the voice selection dialog
        const voiceButton = this.page.locator('button[aria-haspopup="dialog"]').first();
        await voiceButton.waitFor({ state: 'visible', timeout: 5000 });

        // Check if we need to open the dropdown (if not already selected)
        const buttonText = await voiceButton.textContent();
        console.log(`  - Current voice selection: ${buttonText?.trim()}`);

        // If the voice is already selected, skip
        if (buttonText?.toLowerCase().includes(voiceId.toLowerCase())) {
            console.log(`  - Voice ${voiceId} is already selected`);
            console.log(`✅ Voice ID set to: ${voiceId}`);
            return;
        }

        // Click to open the voice selection dropdown
        await voiceButton.click();
        await this.page.waitForTimeout(500);

        // Wait for the dropdown to fully render with visible options
        await this.page.waitForSelector('div:not([hidden])', { timeout: 5000 });

        // Find the voice option by text - look for the voice name in visible elements
        // The dropdown shows options like "Matthew (Male, US)"
        const voiceOptionLocator = this.page.locator(`text="${voiceId}"`).first();

        try {
            await voiceOptionLocator.waitFor({ state: 'visible', timeout: 5000 });
            console.log(`  - Voice option found`);
            await voiceOptionLocator.click();
        } catch {
            // If exact match fails, try partial match with regex
            console.log(`  - Trying partial match for voice: ${voiceId}`);
            const voiceOptions = this.page.locator('div').filter({
                hasText: new RegExp(voiceId, 'i')
            });

            const count = await voiceOptions.count();
            if (count > 0) {
                // Click the last one (likely the voice option, not a section header)
                await voiceOptions.nth(count - 1).click();
                console.log(`  - Selected voice via partial match`);
            } else {
                throw new Error(`Voice option "${voiceId}" not found in dropdown`);
            }
        }

        console.log(`✅ Voice ID set to: ${voiceId}`);
    }

    /**
     * Set the System Prompt in the configuration
     * @param prompt System prompt text to set
     */
    async setSystemPrompt(prompt: string): Promise<void> {
        console.log(`Setting system prompt...`);

        // Find system prompt textarea - it has placeholder "Enter system prompt to define AI behavior..."
        const promptField = this.page.locator('textarea[placeholder*="Enter system prompt"]').first();
        await promptField.waitFor({ state: 'visible', timeout: 5000 });

        // Clear existing content and fill with new prompt
        await promptField.click();
        await promptField.evaluate(el => (el as HTMLTextAreaElement).value = '');
        await promptField.fill(prompt);

        console.log(`✅ System prompt set`);
    }

    /**
     * Set Tool Use configuration
     * @param toolConfig Tool configuration object with tools array
     */
    async setToolUse(toolConfig: { tools: any[] }): Promise<void> {
        console.log(`Setting tool use configuration...`);

        // Tool configuration is a JSON textarea
        const toolConfigField = this.page.locator('textarea[placeholder*="{"]').first();

        const isVisible = await toolConfigField.isVisible().catch(() => false);
        if (isVisible) {
            const configJson = JSON.stringify(toolConfig);
            console.log(`  - Tool configuration: ${configJson}`);

            await toolConfigField.click();
            await toolConfigField.evaluate(el => (el as HTMLTextAreaElement).value = '');
            await toolConfigField.fill(configJson);
            console.log('  - Tool configuration set');
        }

        console.log(`✅ Tool use configured`);
    }

    /**
     * Set Chat History configuration
     * @param chatHistory Chat history array of message objects
     */
    async setChatHistory(chatHistory: any[]): Promise<void> {
        console.log(`Setting chat history...`);

        // Chat history is a JSON textarea field
        // Find the textarea with placeholder containing "[" (JSON array indicator)
        const chatHistoryField = this.page.locator('textarea[placeholder*="[{"]').first();

        const isVisible = await chatHistoryField.isVisible().catch(() => false);
        if (isVisible) {
            const historyJson = JSON.stringify(chatHistory);
            console.log(`  - Chat history: ${historyJson.substring(0, 100)}...`);

            await chatHistoryField.click();
            await chatHistoryField.evaluate(el => (el as HTMLTextAreaElement).value = '');
            await chatHistoryField.fill(historyJson);
            console.log('  - Chat history set');
        }

        console.log(`✅ Chat history configured`);
    }

    /**
     * Set Turn Sensitivity setting
     * @param sensitivity Sensitivity level: 'high', 'medium', or 'low'
     */
    async setTurnSensitivity(sensitivity: 'high' | 'medium' | 'low'): Promise<void> {
        console.log(`Setting turn sensitivity: ${sensitivity}`);

        // Turn sensitivity is a radio button with uppercase value
        const radioValue = sensitivity.toUpperCase();
        const turnSensitivityRadio = this.page.locator(`input[type="radio"][name="turnSensitivity"][value="${radioValue}"]`);

        await turnSensitivityRadio.waitFor({ state: 'visible', timeout: 5000 });
        await turnSensitivityRadio.click();

        console.log(`✅ Turn sensitivity set to: ${sensitivity}`);
    }

    /**
     * Set Sentiment Detection setting
     * @param enabled Whether to enable sentiment detection
     */
    async setSentimentDetection(enabled: boolean): Promise<void> {
        console.log(`Setting sentiment detection: ${enabled}`);

        // Find sentiment detection checkbox/toggle
        const sentimentToggle = this.page.locator('input[type="checkbox"][name*="sentiment" i], [data-testid*="sentiment-detection"], [aria-label*="sentiment" i]').first();

        try {
            await sentimentToggle.waitFor({ state: 'visible', timeout: 3000 });
            const isChecked = await sentimentToggle.isChecked();
            if (isChecked !== enabled) {
                await sentimentToggle.click();
            }
            console.log(`✅ Sentiment detection set to: ${enabled}`);
        } catch {
            console.log('  - Sentiment detection toggle not found, skipping');
        }
    }

    /**
     * Set Inference Configuration settings
     * @param config Inference configuration with maxTokens, temperature, topP
     */
    async setInferenceConfig(config: { maxTokens?: number; temperature?: number; topP?: number }): Promise<void> {
        console.log(`Setting inference configuration...`);

        try {
            // Set Max Tokens if provided
            if (config.maxTokens !== undefined) {
                const maxTokensInput = this.page.locator('input[type="number"][value*="1024"], input[placeholder*="1024"]').first();
                await maxTokensInput.waitFor({ state: 'visible', timeout: 3000 });
                await maxTokensInput.click();
                await maxTokensInput.fill(String(config.maxTokens));
                console.log(`  - Max Tokens: ${config.maxTokens}`);
            }

            // Set Temperature if provided
            if (config.temperature !== undefined) {
                const temperatureInput = this.page.locator('input[type="number"][value*="0.7"], input[step="0.1"]').first();
                await temperatureInput.waitFor({ state: 'visible', timeout: 3000 });
                await temperatureInput.click();
                await temperatureInput.fill(String(config.temperature));
                console.log(`  - Temperature: ${config.temperature}`);
            }

            // Set Top P if provided
            if (config.topP !== undefined) {
                const topPInput = this.page.locator('input[type="number"][value*="0.95"], input[step="0.05"]').first();
                await topPInput.waitFor({ state: 'visible', timeout: 3000 });
                await topPInput.click();
                await topPInput.fill(String(config.topP));
                console.log(`  - Top P: ${config.topP}`);
            }

            console.log(`✅ Inference configuration set`);
        } catch (error) {
            console.log('  - Inference configuration fields not found or already set, skipping');
        }
    }

    /**
     * Set Record Audio setting
     * @param enabled Whether to enable audio recording
     */
    async setRecordAudio(enabled: boolean): Promise<void> {
        console.log(`Setting record audio: ${enabled}`);

        try {
            // Find the record audio checkbox
            const recordAudioCheckbox = this.page.locator('input[type="checkbox"]').filter({ hasText: /enable audio recording/i }).first();

            // Alternative selector if the above doesn't work
            const alternativeCheckbox = this.page.locator('input[type="checkbox"]').nth(1); // Usually the second checkbox

            let checkbox = recordAudioCheckbox;
            const isVisible = await recordAudioCheckbox.isVisible().catch(() => false);

            if (!isVisible) {
                checkbox = alternativeCheckbox;
            }

            await checkbox.waitFor({ state: 'visible', timeout: 3000 });
            const isChecked = await checkbox.isChecked();

            if (isChecked !== enabled) {
                await checkbox.click();
            }

            console.log(`✅ Record audio set to: ${enabled}`);
        } catch (error) {
            console.log('  - Record audio checkbox not found, skipping');
        }
    }

    /**
     * Save the configuration settings
     */
    async saveSettings(): Promise<void> {
        console.log('Saving settings...');

        // Click the "Save Settings" button
        const saveButton = this.page.getByRole('button', { name: /save.*settings|apply/i }).first();
        await saveButton.waitFor({ state: 'visible', timeout: 5000 });
        await saveButton.click();

        // Wait for modal/panel to close or settings to save
        await this.page.waitForTimeout(1000);

        console.log('✅ Settings saved');
    }

    /**
     * Start the conversation by clicking the Start Conversation button
     */
    async startConversation(): Promise<void> {
        console.log('Starting conversation...');

        // Click Start Conversation button
        const startButton = this.page.getByRole('button', { name: /start.*conversation|begin.*conversation|start/i });
        await startButton.click();

        console.log('✅ Conversation started');
    }

    /**
     * Stop the conversation by clicking the Stop Conversation button
     */
    async stopConversation(): Promise<void> {
        console.log('Stopping conversation...');

        // Click Stop Conversation button
        const stopButton = this.page.getByRole('button', { name: /stop.*conversation|end.*conversation|stop/i });
        await stopButton.click();

        console.log('✅ Conversation stopped');
    }

    /**
     * Wait for the conversation to start (WebSocket connection + initial greeting)
     * @param speakFirst If true, wait for bot message before proceeding. If false, wait 3s for app to be ready
     */
    async waitForConversationStart(speakFirst: boolean = true, timeoutMs: number = 45000): Promise<void> {
        console.log(`Waiting for conversation to start (speakFirst: ${speakFirst})...`);

        try {
            if (speakFirst) {
                // speakFirst=true: Wait for the bot to speak first
                console.log('⏳ Waiting for bot message to appear (speakFirst=true)...');
                await this.page.locator('.bot').first().waitFor({ state: 'visible', timeout: timeoutMs });
                console.log('✅ Bot message appeared');
            } else {
                // speakFirst=false: Wait 3s for app recording to initialize and be ready for user audio
                console.log('⏳ Waiting 3s for recording initialization (speakFirst=false)...');
                await this.page.waitForTimeout(3000);
                console.log('✅ Recording initialized - ready for user audio');
            }

            console.log('✅ Conversation started and ready');
        } catch (error) {
            console.error('❌ Error waiting for conversation start:', error);
            throw error;
        }
    }

    /**
     * Setup multiple audio files for microphone input (must be called BEFORE page navigation)
     * @param audioPaths Array of audio file paths to use for each cycle
     */
    async setupMultipleAudioFiles(audioPaths: string[]): Promise<void> {
        console.log(`Setting up ${audioPaths.length} audio files for microphone input...`);

        // Validate all audio files exist
        for (const audioPath of audioPaths) {
            if (!fs.existsSync(audioPath)) {
                throw new Error(`Audio file not found: ${audioPath}`);
            }
        }

        // Read and encode all audio files to base64
        const audioBuffers = audioPaths.map(filePath => {
            const buffer = fs.readFileSync(filePath);
            return buffer.toString('base64');
        });

        // Add init script to set up audio mocking BEFORE page loads
        await this.page.addInitScript((base64Audios: string[]) => {
            // --- AUDIO DETECTION VIA ANALYSER NODE ---
            // This approach analyzes the actual audio signal instead of tracking events

            // Global function to check if audio is currently playing by analyzing frequency data
            (window as any).isAudioPlaying = () => {
                const analyser = (window as any).__audioAnalyser;
                if (!analyser) {
                    return false;
                }

                // Get frequency data from the analyser
                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(dataArray);

                // Check if there's any significant audio signal
                // If sum of frequencies > threshold, audio is playing
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }

                // Threshold: if average frequency > 1, consider it as audio playing
                const average = sum / dataArray.length;
                return average > 1;
            };
            // --- END AUDIO DETECTION ---

            const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(
                navigator.mediaDevices
            );
            let callCount = 0;

            // Store decoded audio buffers for reuse
            const decodedAudioBuffers: AudioBuffer[] = [];
            let audioContext: AudioContext | null = null;

            navigator.mediaDevices.getUserMedia = async function (constraints: any) {
                if (constraints.audio) {
                    try {
                        callCount++;
                        (window as any).__lastGetUserMediaCallNumber = callCount;
                        console.log(`🎤 getUserMedia call #${callCount}`);

                        // Create audio context if not already created
                        if (!audioContext) {
                            audioContext = new (window.AudioContext ||
                                (window as any).webkitAudioContext)();
                            (window as any).__audioContext = audioContext;
                            console.log('🎤 Audio context created');

                            // Create analyser node for audio detection
                            const analyser = audioContext.createAnalyser();
                            analyser.fftSize = 2048;
                            analyser.connect(audioContext.destination);
                            (window as any).__audioAnalyser = analyser;
                            console.log('🎧 Audio analyser created and connected');
                        }

                        // Decode audio buffers on first call
                        if (decodedAudioBuffers.length === 0) {
                            console.log(`🎤 Decoding ${base64Audios.length} audio files...`);
                            for (let i = 0; i < base64Audios.length; i++) {
                                const binaryString = atob(base64Audios[i]);
                                const bytes = new Uint8Array(binaryString.length);
                                for (let j = 0; j < binaryString.length; j++) {
                                    bytes[j] = binaryString.charCodeAt(j);
                                }
                                const decodedBuffer = await audioContext.decodeAudioData(
                                    bytes.buffer
                                );
                                decodedAudioBuffers.push(decodedBuffer);
                            }
                            (window as any).__audioBuffers = decodedAudioBuffers;
                            console.log(
                                `🎤 ${decodedAudioBuffers.length} audio buffers created and stored`
                            );
                        }

                        // CRITICAL: Always create a NEW destination for each getUserMedia call
                        const destination = audioContext.createMediaStreamDestination();
                        (window as any).__audioDestination = destination;
                        (window as any).__audioDestinationCallNumber = callCount;
                        console.log(`🎤 Created new destination for call #${callCount}`);

                        return destination.stream;
                    } catch (error) {
                        console.error('Failed to create custom audio stream:', error);
                        return originalGetUserMedia(constraints);
                    }
                }
                return originalGetUserMedia(constraints);
            };

            // Implement triggerTestAudio to support multiple audio buffers
            (window as any).triggerTestAudio = (cycleIndex: number = 0) => {
                console.log(`\n🎵 triggerTestAudio called with cycleIndex=${cycleIndex}`);
                const context = (window as any).__audioContext;
                const audioBuffers = (window as any).__audioBuffers;
                const destination = (window as any).__audioDestination;
                const destCallNumber = (window as any).__audioDestinationCallNumber;
                const lastCallNumber = (window as any).__lastGetUserMediaCallNumber;

                console.log(`📋 State check:`);
                console.log(`   - context: ${!!context}`);
                console.log(`   - audioBuffers: ${!!audioBuffers}, length: ${audioBuffers?.length || 0}`);
                console.log(`   - destination: ${!!destination}`);
                console.log(`   - destCallNumber: ${destCallNumber}`);
                console.log(`   - lastCallNumber: ${lastCallNumber}`);

                if (!context || !audioBuffers || !destination || audioBuffers.length === 0) {
                    console.error(
                        '❌ Missing context, buffers, or destination for audio playback'
                    );
                    return false;
                }

                // Ensure cycleIndex is valid
                if (cycleIndex < 0 || cycleIndex >= audioBuffers.length) {
                    console.error(
                        `❌ Invalid cycle index ${cycleIndex}, available buffers: ${audioBuffers.length}`
                    );
                    return false;
                }

                // If destination is stale, return false
                if (destCallNumber !== lastCallNumber) {
                    console.error(
                        `❌ Destination stale (call #${destCallNumber} vs latest #${lastCallNumber})`
                    );
                    return false;
                }

                try {
                    // Check if audio context is still running
                    if (context.state !== 'running') {
                        console.error(`❌ AudioContext not running: ${context.state}`);
                        return false;
                    }

                    // Check if destination stream is actually active
                    const stream = destination.stream;
                    const audioTracks = stream.getAudioTracks();
                    console.log(`📋 Audio track info: tracks=${audioTracks.length}, readyState=${audioTracks[0]?.readyState}`);

                    if (audioTracks.length === 0 || audioTracks[0].readyState !== 'live') {
                        console.error(
                            `❌ Audio track not live: tracks=${audioTracks.length}, readyState=${audioTracks[0]?.readyState}`
                        );
                        return false;
                    }

                    // Get the audio buffer for this cycle
                    const buffer = audioBuffers[cycleIndex];
                    console.log(`📋 Buffer info: index=${cycleIndex}, duration=${buffer?.duration}s, sampleRate=${buffer?.sampleRate}Hz`);

                    // Create a new source each time (old one cannot be restarted)
                    const source = context.createBufferSource();
                    source.buffer = buffer;

                    // Get the analyser node
                    const analyser = (window as any).__audioAnalyser;

                    // Connect to both the destination stream AND the analyser for monitoring
                    source.connect(destination);
                    if (analyser) {
                        source.connect(analyser);
                    }

                    // Start playing from the beginning
                    source.start(0);

                    // Track audio activity: record when this audio playback started and calculate when it will end
                    const bufferDuration = buffer.duration;
                    const startTime = context.currentTime;
                    const endTime = startTime + bufferDuration;

                    (window as any).__lastAudioStartTime = startTime;
                    (window as any).__lastAudioEndTime = endTime;

                    console.log(`🎵 Audio started playing at context time ${startTime}s (duration: ${bufferDuration}s, will end at: ${endTime}s)`);

                    console.log(`✅ Audio response #${cycleIndex + 1} triggered (using buffer ${cycleIndex})`);
                    return true;
                } catch (error) {
                    console.error('❌ Audio trigger failed:', error);
                    return false;
                }
            };
        }, audioBuffers);

        console.log(`✅ Audio files setup complete: ${audioPaths.length} files loaded`);
    }

    /**
     * Setup audio completion listener
     * NOTE: Audio completion is now detected via console.log capture in setupMultipleAudioFiles()
     */
    setupAudioCompletionListener(): void {
        console.log('✅ Audio completion detection initialized (via addInitScript console capture)');
    }

    /**
     * Wait for silence (no browser audio playback) for a specified duration
     * @param silenceDurationMs Required duration of silence in milliseconds (default 5000ms)
     * @param timeoutMs Maximum time to wait for silence
     * @param returnImmediatelyIfSilent If true, returns immediately if already silent (for initial check)
     * @returns true if silence detected, false if timeout
     */
    private async waitForSilence(
        silenceDurationMs: number = 5000,
        timeoutMs: number = 60000,
        returnImmediatelyIfSilent: boolean = false
    ): Promise<boolean> {
        const startTime = Date.now();
        let silenceStartTime: number | null = null;

        console.log(`  👂 Listening for ${silenceDurationMs}ms of silence (timeout: ${timeoutMs}ms)...`);

        while (Date.now() - startTime < timeoutMs) {
            const audioStatus = await this.page.evaluate(() => {
                // Use analyser to detect actual audio signal
                const isPlaying = (window as any).isAudioPlaying();

                // Check the mock input specifically (User Mic)
                const context = (window as any).__audioContext;
                const lastEndTime = (window as any).__lastAudioEndTime;

                let mockInputPlaying = false;
                if (context && lastEndTime) {
                    mockInputPlaying = context.currentTime < lastEndTime;
                }

                // Audio is playing if EITHER the app is outputting sound OR the mock mic is running
                const audioPlaying = isPlaying || mockInputPlaying;

                return {
                    audioPlaying: audioPlaying,
                    reason: isPlaying ? 'signal detected' : (mockInputPlaying ? 'mock mic' : 'silence')
                };
            });

            if (!audioStatus.audioPlaying) {
                // Audio is silent
                if (silenceStartTime === null) {
                    silenceStartTime = Date.now();
                    console.log(`  🔇 Silence started (${audioStatus.reason})`);

                    // If we should return immediately when silent, do so
                    if (returnImmediatelyIfSilent) {
                        console.log(`  ✅ Already silent - returning immediately`);
                        return true;
                    }
                }

                const silenceDuration = Date.now() - silenceStartTime;
                if (silenceDuration >= silenceDurationMs) {
                    console.log(`  ✅ ${silenceDurationMs}ms of silence detected`);
                    // Reset the audio completed flag for next cycle
                    await this.page.evaluate(() => {
                        (window as any).__audioPlaybackCompleted = false;
                    });
                    return true;
                }
            } else {
                // Audio is playing - reset silence timer
                if (silenceStartTime !== null) {
                    console.log(`  🔊 Audio playing again (${audioStatus.reason})`);
                }
                silenceStartTime = null;
            }

            await this.page.waitForTimeout(200);
        }

        console.warn(`  ⏱️ Timeout waiting for ${silenceDurationMs}ms of silence (${timeoutMs}ms elapsed)`);
        return false;
    }

    /**
     * Check if audio has been idle for at least 5 seconds
     * @param timeoutMs Maximum time to wait for idle state
     * @returns true if 5+ seconds of no audio activity detected, false if timeout
     * @deprecated Use waitForSilence() instead
     */
    private async waitForAudioIdle(timeoutMs: number = 60000): Promise<boolean> {
        const startTime = Date.now();

        while (Date.now() - startTime < timeoutMs) {
            const idleInfo = await this.page.evaluate(() => {
                const context = (window as any).__audioContext;
                const lastEndTime = (window as any).__lastAudioEndTime;
                const audioCompleted = (window as any).__audioPlaybackCompleted;

                if (!context || lastEndTime === undefined) {
                    return { isIdle: false, reason: 'no context or lastEndTime' };
                }

                const currentTime = context.currentTime;
                const timeSinceUserAudio = currentTime - lastEndTime;

                // Check if assistant's audio playback has completed AND we're past user audio
                if (audioCompleted && timeSinceUserAudio >= 2) {
                    // Reset the flag so next cycle can detect completion
                    (window as any).__audioPlaybackCompleted = false;
                    return { isIdle: true, reason: 'audio completed' };
                }

                // Fallback: If we're at least 7 seconds past the user audio end time, assume we're idle
                // (This handles cases where audio completion event doesn't fire)
                if (timeSinceUserAudio >= 7) {
                    return { isIdle: true, reason: 'timeout based idle (7s)' };
                }

                return { isIdle: false, reason: `waiting (${timeSinceUserAudio.toFixed(1)}s since user audio)` };
            });

            if (idleInfo.isIdle) {
                console.log(`  - Audio idle detected: ${idleInfo.reason}`);
                return true;
            }

            await this.page.waitForTimeout(300);
        }

        console.warn(`⏱️ Timeout waiting for audio idle (${timeoutMs}ms)`);
        return false;
    }

    /**
     * Wait for AI audio playback to complete via console event detection
     * @param timeoutMs Maximum time to wait
     * @returns true if audio completed, false if timeout
     */
    private async waitForAudioPlaybackViaEvents(timeoutMs: number = 45000): Promise<boolean> {
        const startTime = Date.now();
        let lastLogTime = startTime;
        const logIntervalMs = 5000;

        console.log('🎧 Waiting for AI audio playback to complete (via events)...');

        // Reset the flag before waiting
        await this.page.evaluate(() => {
            (window as any).__audioPlaybackCompleted = false;
        });

        while (Date.now() - startTime < timeoutMs) {
            // Check if audio completion flag was set
            const completed = await this.page.evaluate(() => {
                return (window as any).__audioPlaybackCompleted === true;
            });

            if (completed) {
                console.log('✅ AI audio playback completed (detected via console event)');
                return true;
            }

            // Show progress every 5 seconds
            const now = Date.now();
            if (now - lastLogTime >= logIntervalMs) {
                const elapsedMs = now - startTime;
                const remainingMs = timeoutMs - elapsedMs;
                console.log(
                    `⏳ Waiting for audio completion... (${Math.ceil(remainingMs / 1000)}s remaining)`
                );
                lastLogTime = now;
            }

            await this.page.waitForTimeout(300);
        }

        console.warn(`⏱️ Timeout waiting for AI audio playback via events (${timeoutMs}ms)`);
        return false;
    }

    /**
     * Get count of FINAL messages by role
     * Counts div.bot elements with [FINAL] for assistant
     * Counts user messages using generic selector
     */
    private async getFinalMessageCounts(): Promise<{ assistantCount: number; userCount: number }> {
        try {
            const counts = await this.page.evaluate(() => {
                let assistantCount = 0;
                let userCount = 0;

                // Count bot messages with [FINAL] marker
                const botMessages = document.querySelectorAll('div.bot');
                botMessages.forEach((msg) => {
                    if (msg.textContent && msg.textContent.includes('[FINAL]')) {
                        assistantCount++;
                    }
                });

                // Count user messages using generic selector
                const userMessages = document.querySelectorAll(
                    '[data-testid*="user"], .user-message, [class*="user"]'
                ).length;
                userCount = userMessages;

                return {
                    assistantCount,
                    userCount,
                };
            });

            return counts;
        } catch (error) {
            console.error('Error getting message counts:', error);
            return { assistantCount: 0, userCount: 0 };
        }
    }

    /**
     * Trigger audio playback - automatically detects if assistant speaks first
     * Flow: Monitor for 5s → Play first audio → Wait 5s silence → Play next audio → Repeat
     * @param numAudioFiles Number of audio files to play
     * @param maxWaitPerCycle Maximum time to wait for silence between audio files (milliseconds)
     * @returns Object with success status
     */
    async triggerAudioPlayback(
        numAudioFiles: number = 1,
        maxWaitPerCycle: number = 240000,
        speakFirst: boolean = false  // Ignored - auto-detected
    ): Promise<{ success: boolean }> {
        console.log(`🎤 Starting audio playback for ${numAudioFiles} files...`);

        // Verify triggerTestAudio exists
        const triggerTestAudioExists = await this.page.evaluate(() => {
            return typeof (window as any).triggerTestAudio === 'function';
        });

        if (!triggerTestAudioExists) {
            console.error('❌ triggerTestAudio function not found! Audio mocking may not be set up correctly.');
            return { success: false };
        }

        // Step 1: Wait 5 seconds after conversation start to detect if assistant speaks first
        console.log(`⏳ Monitoring for 5 seconds to detect if assistant speaks first...`);
        const assistantSpokeFirst = await this.waitForSilence(5000, 5000, true);

        if (!assistantSpokeFirst) {
            console.log(`🎯 Assistant spoke first - waiting for response to complete...`);
            // Wait for assistant's audio to finish
            await this.waitForSilence(5000, maxWaitPerCycle);
            console.log(`✅ Assistant audio complete`);
        } else {
            console.log(`🎯 No audio detected - user speaks first`);
        }

        // Step 2: Play all audio files sequentially with 5-second silence detection between them
        for (let audioIndex = 0; audioIndex < numAudioFiles; audioIndex++) {
            console.log(`\n🔄 ════════════════════════════════════════`);
            console.log(`🔄 Audio ${audioIndex + 1}/${numAudioFiles}`);

            // Trigger the test audio
            console.log(`🎤 Playing test audio ${audioIndex + 1}...`);
            const triggered = await this.page.evaluate((index: number) => {
                if (typeof (window as any).triggerTestAudio !== 'function') {
                    return false;
                }
                try {
                    return (window as any).triggerTestAudio(index);
                } catch (error) {
                    console.error('❌ Failed to trigger audio:', error);
                    return false;
                }
            }, audioIndex);

            if (!triggered) {
                console.warn(`⚠️ Failed to trigger audio ${audioIndex + 1}`);
                return { success: false };
            }

            console.log(`✅ Test audio ${audioIndex + 1} triggered`);

            // Wait for 5 seconds of silence before next audio
            console.log(`⏳ Waiting for 5 seconds of silence...`);
            const silenceDetected = await this.waitForSilence(5000, maxWaitPerCycle);

            if (!silenceDetected) {
                console.warn(`⚠️ Timeout waiting for silence after audio ${audioIndex + 1}`);
                return { success: false };
            }

            console.log(`✅ 5 seconds of silence detected`);
        }

        console.log(`\n✅ All ${numAudioFiles} audio files completed successfully!`);
        return { success: true };
    }

    /**
     * Wait for a specific message count to be reached
     * @param role Message role to count ('assistant' or 'user')
     * @param expectedCount Expected number of FINAL messages
     * @param timeoutMs Maximum time to wait in milliseconds
     * @returns true if count reached, false if timeout
     */
    private async waitForMessageCount(
        role: 'assistant' | 'user',
        expectedCount: number,
        timeoutMs: number = 60000
    ): Promise<boolean> {
        const startTime = Date.now();
        const roleLabel = role === 'assistant' ? 'AI' : 'User';

        while (Date.now() - startTime < timeoutMs) {
            const counts = await this.getFinalMessageCounts();
            const currentCount = role === 'assistant' ? counts.assistantCount : counts.userCount;

            if (currentCount >= expectedCount) {
                return true;
            }

            // Check every 500ms
            await this.page.waitForTimeout(500);
        }

        console.warn(`⏱️ Timeout waiting for ${roleLabel} message count`);

        return false;
    }

    /**
     * Check if conversation is completed
     */
    async isConversationCompleted(): Promise<boolean> {
        try {
            const stopButton = this.page.getByRole('button', { name: /stop.*conversation|end/i });
            return !(await stopButton.isVisible({ timeout: 1000 }));
        } catch {
            return true;
        }
    }

    /**
     * Wait for conversation completion
     */
    async waitForConversationCompletion(timeoutMs: number = 30000): Promise<boolean> {
        try {
            const stopButton = this.page.getByRole('button', { name: /stop.*conversation|end/i });
            await stopButton.waitFor({ state: 'hidden', timeout: timeoutMs });
            return true;
        } catch {
            return false;
        }
    }
}
