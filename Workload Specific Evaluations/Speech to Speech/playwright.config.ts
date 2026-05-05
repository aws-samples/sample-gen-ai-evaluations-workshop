import { defineConfig, devices } from '@playwright/test';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.test' });
dotenv.config({ path: '.env.test.local', override: true });

/**
 * Playwright configuration for Interview Assistant E2E tests
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
    testDir: './test/e2e',

    /* Run tests in files in parallel */
    fullyParallel: false,

    /* Fail the build on CI if you accidentally left test.only in the source code */
    forbidOnly: !!process.env.CI,

    /* Retry on CI only */
    retries: process.env.CI ? 1 : 0,

    /* Opt out of parallel tests on CI */
    workers: process.env.CI ? 1 : undefined,

    /* Reporter to use */
    reporter: [
        ['html'],
        ['list'],
        ['json', { outputFile: 'test-results/results.json' }]
    ],

    /* Shared settings for all the projects below */
    use: {
        /* Base URL to use in actions like `await page.goto('/')` */
        baseURL: process.env.FRONTEND_URL || 'http://localhost:3000',

        /* Collect trace when retrying the failed test */
        trace: 'on-first-retry',

        /* Screenshot on failure */
        screenshot: 'only-on-failure',

        /* Video on failure */
        video: 'retain-on-failure',

        /* Maximum time each action can take */
        actionTimeout: 15000,

        /* Maximum time for navigation */
        navigationTimeout: 30000,

        /* Grant microphone permissions for audio tests */
        permissions: ['microphone'],
    },

    /* Configure timeout */
    timeout: 60000,
    expect: {
        timeout: 10000,
    },

    /* Configure projects for major browsers */
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },

        // {
        //     name: 'firefox',
        //     use: { ...devices['Desktop Firefox'] },
        // },

        // {
        //     name: 'webkit',
        //     use: { ...devices['Desktop Safari'] },
        // },

        // /* Test against mobile viewports */
        // {
        //     name: 'Mobile Chrome',
        //     use: { ...devices['Pixel 5'] },
        // },
        // {
        //     name: 'Mobile Safari',
        //     use: { ...devices['iPhone 12'] },
        // },
    ],

    /* Run local dev server before starting the tests */
    // Note: webServer disabled - manually start backend and frontend before running tests
    // Backend: cd lib/stacks/backend/app && python run_servers.py
    // Frontend: npm run dev --workspace=lib/stacks/frontend/app
    // webServer: process.env.CI ? undefined : [
    //     {
    //         command: 'cd lib/stacks/backend/app && python run_servers.py',
    //         url: 'http://localhost:8000',
    //         reuseExistingServer: !process.env.CI,
    //         timeout: 120000,
    //     },
    //     {
    //         command: 'npm run dev --workspace=lib/stacks/frontend/app',
    //         url: 'http://localhost:3000',
    //         reuseExistingServer: !process.env.CI,
    //         timeout: 120000,
    //     },
    // ],
});
