import { test, expect } from '@playwright/test';
import { TestHelper } from './helpers/test.helper';
import { scanTestScenarios, TestScenario } from './helpers/test-data.helper';
import * as path from 'path';
import * as fs from 'fs';
import * as dotenv from 'dotenv';

// Load test environment variables
// Load .env.test first (defaults), then .env.test.local (overrides)
dotenv.config({ path: '.env.test' });
dotenv.config({ path: '.env.test.local', override: true });

/**
 * Environment Variables:
 * - ENABLE_AUDIO_TESTS: Enable/disable audio tests (true | false, default: false)
 * - FRONTEND_URL: URL of the S2S application (default: http://localhost:3000)
 */

// Test configuration
const TEST_CONFIG = {
    testDataRoot: process.env.TEST_DATA_ROOT || path.resolve(__dirname, '../../test-data'),
    enableAudioTests: process.env.ENABLE_AUDIO_TESTS === 'true',
    frontendUrl: process.env.FRONTEND_URL || 'http://localhost:3000'
};

// Scan test scenarios dynamically
const testScenarios = TEST_CONFIG.enableAudioTests ? scanTestScenarios(TEST_CONFIG.testDataRoot) : [];

test.describe('Speech-to-Speech Conversation - End-to-End Flow with Dynamic Test Scenarios', () => {
    // Generate one test per scenario
    testScenarios.forEach((scenario: TestScenario) => {
        test.describe(`Scenario: ${scenario.name}`, () => {
            let testHelper: TestHelper;

            test.beforeEach(async ({ page, context }) => {
                testHelper = new TestHelper(page);

                // Grant microphone permissions for audio tests
                await context.grantPermissions(['microphone']);

                // Verify all audio files for this scenario exist
                for (const audioPath of scenario.audioFilePaths) {
                    if (!fs.existsSync(audioPath)) {
                        throw new Error(`Audio file not found at: ${audioPath}`);
                    }
                }

                // Setup audio BEFORE any page navigation (must be done before page loads)
                await testHelper.setupMultipleAudioFiles(scenario.audioFilePaths);

                console.log(`✅ Scenario "${scenario.name}" loaded`);
                console.log(`   - Profile: ${scenario.config.profile}`);
                console.log(`   - Voice ID: ${scenario.config.voiceId}`);
                console.log(`   - Audio files: ${scenario.audioFilePaths.length}`);
            });

            test('Complete S2S conversation flow - Start to End', async ({ page }) => {
                const numCycles = scenario.audioFilePaths.length;
                test.setTimeout(900000 + numCycles * 240000); // Base 15 min + 4 min per cycle

                // Step 1: Navigate to application home
                await test.step('Navigate to S2S application', async () => {
                    await page.goto(TEST_CONFIG.frontendUrl || '/');
                    await page.waitForLoadState('networkidle');
                    console.log('✅ Application loaded');
                });

                // Step 2: Select profile from dropdown
                await test.step('Select profile from dropdown', async () => {
                    await testHelper.selectProfile(scenario.config.profile);
                });

                // Step 3: Open and configure settings
                await test.step('Configure conversation settings', async () => {
                    await testHelper.openConfigSettings();
                    await testHelper.setVoiceId(scenario.config.voiceId);
                    await testHelper.setSystemPrompt(scenario.config.systemPrompt);

                    // Set tool use configuration from config
                    if (scenario.config.toolUse) {
                        await testHelper.setToolUse(scenario.config.toolUse);
                    }

                    // Set chat history configuration from config
                    if (scenario.config.chatHistory !== undefined) {
                        await testHelper.setChatHistory(scenario.config.chatHistory);
                    }

                    // Set turn sensitivity if configured
                    if (scenario.config.turnSensitivity) {
                        await testHelper.setTurnSensitivity(scenario.config.turnSensitivity);
                    }

                    // Set sentiment detection if configured
                    if (scenario.config.sentimentDetection !== undefined) {
                        await testHelper.setSentimentDetection(scenario.config.sentimentDetection);
                    }

                    // Set inference configuration if configured
                    if (scenario.config.inferenceConfig) {
                        await testHelper.setInferenceConfig(scenario.config.inferenceConfig);
                    }

                    // Set audio recording if configured
                    if (scenario.config.saveAudio !== undefined) {
                        await testHelper.setRecordAudio(scenario.config.saveAudio);
                    }

                    await testHelper.saveSettings();
                });

                // Step 4: Start the conversation
                await test.step('Start conversation', async () => {
                    await testHelper.startConversation();
                    // Just wait for WebSocket to connect and recording to start
                    await page.waitForTimeout(3000);
                });

                // Step 5: Conduct audio playback (auto-detects who speaks first)
                await test.step(`Play ${numCycles} audio files`, async () => {
                    console.log(`Starting playback of ${numCycles} audio files...`);

                    // Setup audio completion listener
                    testHelper.setupAudioCompletionListener();

                    // Trigger audio playback - automatically detects if assistant speaks first
                    const result = await testHelper.triggerAudioPlayback(
                        numCycles,
                        240000  // 240s max per audio file
                    );

                    if (result.success) {
                        console.log(`✅ All audio files played successfully!`);
                    } else {
                        console.warn('⚠️ Audio playback did not complete successfully');
                    }

                    // Wait a moment for final responses to settle
                    await page.waitForTimeout(3000);
                });

                // Step 6: Stop the conversation
                await test.step('Stop conversation', async () => {
                    await testHelper.stopConversation();
                    await testHelper.waitForConversationCompletion();
                    console.log('✅ Conversation ended successfully');
                });
            });
        });
    });

    // Fallback: No test scenarios found
    if (testScenarios.length === 0) {
        test('No test scenarios configured', async () => {
            test.skip(
                true,
                'No test scenarios found in test_data directory. Audio tests are disabled or no scenarios are configured.'
            );
        });
    }
});
