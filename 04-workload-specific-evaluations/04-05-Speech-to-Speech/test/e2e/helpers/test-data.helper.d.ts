export interface TestScenario {
    name: string;
    folderPath: string;
    audioFilePaths: string[];
}
/**
 * Scans the test_data directory for test scenarios
 * Each subdirectory is a test scenario containing numerically-prefixed .wav files
 * @param testDataRoot Root path to test_data folder (e.g., './test_data')
 * @returns Array of test scenarios sorted by folder name
 */
export declare function scanTestScenarios(testDataRoot: string): TestScenario[];
/**
 * Get a single test scenario by name
 * @param testDataRoot Root path to test_data folder
 * @param scenarioName Name of the scenario to retrieve
 * @returns Test scenario or null if not found
 */
export declare function getTestScenario(testDataRoot: string, scenarioName: string): TestScenario | null;
