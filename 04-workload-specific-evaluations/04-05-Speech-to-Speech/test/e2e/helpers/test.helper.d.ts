import { Page } from '@playwright/test';
export declare class TestHelper {
    private page;
    constructor(page: Page);
    /**
     * Upload a file (job description or CV)
     */
    uploadFile(filePath: string): Promise<void>;
    /**
     * Upload CV file (legacy method, use uploadFile instead)
     */
    uploadCV(cvPath: string): Promise<void>;
    /**
     * Enter job description
     */
    enterJobDescription(jobDescription: string): Promise<void>;
    /**
     * Enter job posting URL
     */
    enterJobUrl(url: string): Promise<void>;
    /**
     * Enter LinkedIn profile URL or CV URL
     */
    enterProfileUrl(url: string): Promise<void>;
    /**
     * Save interview preparation
     */
    savePreparation(): Promise<void>;
    /**
     * Start practice session (from Live Practice page)
     */
    startPracticeSession(): Promise<void>;
    /**
     * Select first preparation and continue
     */
    selectFirstPreparationAndContinue(): Promise<void>;
    /**
     * Configure interview settings (Light/Smart Mode + Audio Recording)
     * Mode selection is determined by INTERVIEW_MODE environment variable:
     * - 'light' (default): Selects Light Mode
     * - 'smart': Selects Smart Mode
     */
    configureInterviewSettings(lightMode?: boolean, saveAudioRecording?: boolean): Promise<void>;
    /**
     * Start the live practice session (clicks the Start button)
     */
    startLivePracticeSession(): Promise<void>;
    /**
     * Start audio playback after a delay (simulates user response)
     * @param delayMs Delay in milliseconds before triggering audio
     */
    startAudioPlaybackAfterDelay(delayMs?: number): Promise<void>;
    /**
     * Start practice now (navigates to Live Practice page)
     */
    startPracticeNow(): Promise<void>;
    /**
     * Start interview session (legacy - now calls savePreparation then startPracticeNow)
     */
    startInterview(): Promise<void>;
    /**
     * Wait for WebSocket connection
     */
    waitForWebSocketConnection(timeout?: number): Promise<void>;
    /**
     * Send text message in chat
     */
    sendMessage(message: string): Promise<void>;
    /**
     * Wait for AI response
     */
    waitForAIResponse(timeout?: number): Promise<string>;
    /**
     * Setup multiple audio files for microphone input
     * Must be called BEFORE navigating to the page that uses getUserMedia
     * @param audioPaths Array of audio file paths to use for each cycle
     */
    setupMultipleAudioFiles(audioPaths: string[]): Promise<void>;
    /**
     * Setup audio file as microphone input (legacy, uses first audio file)
     * Must be called BEFORE navigating to the page that uses getUserMedia
     * @param audioPath Path to the audio file to use as microphone input
     */
    setupAudioFile(audioPath: string): Promise<void>;
    /**
     * Setup audio completion detection
     * NOTE: Audio completion is now detected via console.log capture in setupAudioFile()
     * This method is kept for compatibility but no longer uses page.on('console') listeners
     * to avoid async/await issues on subsequent test runs
     */
    setupAudioCompletionListener(): void;
    /**
     * Wait for AI audio playback to complete via console event detection
     * Listens for "[AudioPlayer] All audio playback completed, calling callback" message
     * @param timeoutMs Maximum time to wait
     * @returns true if audio completed, false if timeout
     */
    private waitForAudioPlaybackViaEvents;
    /**
     * Get count of FINAL messages by role
     */
    private getFinalMessageCounts;
    /**
     * Conduct multiple question-answer cycles with audio playback
     * Handles the full interview flow with multiple Q&A rounds
     * @param numCycles Number of question-answer cycles to complete (default: 3)
     * @param maxWaitPerCycle Maximum time to wait per cycle (default: 45000ms / 45 seconds)
     * @returns Object with success status and final message counts
     */
    triggerAudioPlayback(numCycles?: number, maxWaitPerCycle?: number): Promise<{
        success: boolean;
        assistantCount: number;
        userCount: number;
    }>;
    /**
     * Wait for a specific message count to be reached
     * @param role Message role to count ('assistant' or 'user')
     * @param expectedCount Expected number of FINAL messages
     * @param timeoutMs Maximum time to wait in milliseconds
     * @returns true if count reached, false if timeout
     */
    private waitForMessageCount;
    /**
     * Check for interview completion
     */
    isInterviewCompleted(): Promise<boolean>;
    /**
     * Wait for feedback to be available on the session details page
     * Backend analysis may still be running, so poll until it completes or timeout
     * @param timeoutMs Maximum time to wait
     * @returns true if feedback is available, false if timeout
     */
    waitForFeedbackAvailability(timeoutMs?: number): Promise<boolean>;
    /**
     * Get feedback summary from the Practice Session Details page
     * Assumes we're already on the session details page (/practice/session/{id})
     *
     * The page structure is:
     * - Interview Feedback section with h2 heading
     * - Overall Score
     * - Summary
     * - Key Strengths
     * - Areas for Improvement
     * - Detailed Evaluation (expandable)
     */
    getFeedbackSummary(): Promise<string | null>;
}
