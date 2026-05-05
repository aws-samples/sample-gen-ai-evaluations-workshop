import * as fs from 'fs';
import * as path from 'path';

export interface ScenarioConfig {
    profile: string;
    voiceId: string;
    systemPrompt: string;
    toolUse?: { tools: any[] };
    chatHistory?: any[];
    turnSensitivity?: 'high' | 'medium' | 'low';
    sentimentDetection?: boolean;
    saveAudio?: boolean;
    inferenceConfig?: {
        maxTokens?: number;
        temperature?: number;
        topP?: number;
    };
}

export interface TestScenario {
    name: string;
    folderPath: string;
    audioFilePaths: string[];
    config: ScenarioConfig;
}

/**
 * Loads and validates a scenario config file
 * @param scenarioPath Path to the scenario folder
 * @returns Parsed config or default config if file doesn't exist
 */
function loadScenarioConfig(scenarioPath: string): ScenarioConfig {
    const configPath = path.join(scenarioPath, 'config.json');

    const defaultConfig: ScenarioConfig = {
        profile: 'default',
        voiceId: 'default',
        systemPrompt: 'You are a helpful assistant.',
        toolUse: { tools: [] },
        chatHistory: [],
        speakFirst: false,
    };

    if (!fs.existsSync(configPath)) {
        console.warn(`⚠️ Config file not found at ${configPath}, using default config`);
        return defaultConfig;
    }

    try {
        const configData = fs.readFileSync(configPath, 'utf-8');
        const config = JSON.parse(configData) as ScenarioConfig;

        // Validate required fields
        if (!config.profile || !config.voiceId || !config.systemPrompt) {
            console.warn(`⚠️ Config missing required fields in ${configPath}, using defaults for missing fields`);
            return { ...defaultConfig, ...config };
        }

        return config;
    } catch (error) {
        console.error(`❌ Error parsing config file at ${configPath}:`, error);
        return defaultConfig;
    }
}

/**
 * Scans the test_data directory for test scenarios
 * Each subdirectory is a test scenario containing numerically-prefixed .wav files and a config.json
 * @param testDataRoot Root path to test_data folder (e.g., './test_data')
 * @returns Array of test scenarios sorted by folder name
 */
export function scanTestScenarios(testDataRoot: string): TestScenario[] {
    const scenarios: TestScenario[] = [];

    if (!fs.existsSync(testDataRoot)) {
        console.warn(`Test data root not found: ${testDataRoot}`);
        return scenarios;
    }

    // Read all directories in test_data
    const entries = fs.readdirSync(testDataRoot, { withFileTypes: true });
    const subdirs = entries.filter(entry => entry.isDirectory()).map(entry => entry.name);

    // Sort subdirectories alphabetically
    subdirs.sort();

    // Process each subdirectory as a test scenario
    subdirs.forEach(subdir => {
        const scenarioPath = path.join(testDataRoot, subdir);
        const audioFiles = fs
            .readdirSync(scenarioPath)
            .filter(file => file.endsWith('.wav'))
            .sort((a, b) => {
                // Extract numeric prefix and sort numerically
                const numA = parseInt(a.split('_')[0], 10);
                const numB = parseInt(b.split('_')[0], 10);
                return numA - numB;
            });

        if (audioFiles.length > 0) {
            const audioFilePaths = audioFiles.map(file => path.resolve(path.join(scenarioPath, file)));
            const config = loadScenarioConfig(scenarioPath);

            scenarios.push({
                name: subdir,
                folderPath: scenarioPath,
                audioFilePaths,
                config,
            });

            console.log(`✅ Test scenario detected: ${subdir} (${audioFiles.length} audio files, profile: ${config.profile})`);
        } else {
            console.warn(`⚠️ Skipping ${subdir} - no .wav files found`);
        }
    });

    console.log(`📊 Total test scenarios found: ${scenarios.length}`);
    return scenarios;
}

/**
 * Get a single test scenario by name
 * @param testDataRoot Root path to test_data folder
 * @param scenarioName Name of the scenario to retrieve
 * @returns Test scenario or null if not found
 */
export function getTestScenario(testDataRoot: string, scenarioName: string): TestScenario | null {
    const scenarios = scanTestScenarios(testDataRoot);
    return scenarios.find(s => s.name === scenarioName) || null;
}
