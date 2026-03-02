# E2E Tests - Speech-to-Speech Conversation

End-to-end tests for the Speech-to-Speech (Nova Sonic) applications using Playwright.

## Overview

This test suite covers the complete user journey through a Speech-to-Speech (S2S) conversation application, showcasing:

- ✅ Profile selection from dropdown
- ✅ Configuration settings (Voice ID, System Prompt, Tool Use, Chat History)
- ✅ Start/Stop Conversation
- ✅ Audio interaction with WebSocket streaming
- ✅ Multi-cycle Q&A with audio files
- ✅ CloudWatch trace telemetry (for evaluation pipeline)

The test scenarios are configuration-driven, with each sub-directory in `test_data` containing:
- `config.json` - Profile and settings configuration
- `*.wav` files - Audio files for conversation cycles (numbered: `1_*.wav`, `2_*.wav`, etc.)

## Prerequisites

1. **Node.js 22+** installed
2. **Playwright browsers** installed:
   ```bash
   npx playwright install
   ```
3. **Application deployed** or **running locally** (see instructions in [sample_s2s_app README.md](../../sample_s2s_app/README.md))
4. **Test environment configured** (see Configuration section)

## Configuration

### 1. Create `.env.test`

Copy `env.example` to `.env.test` and configure:

```bash
cp env.example .env.test
```

### 2. Update Environment Variables (if needed)

Edit `.env.test:

```bash
# for local development
FRONTEND_URL=http://localhost:5173

# Enable audio tests
ENABLE_AUDIO_TESTS=true
```

### 2. Copy to test folder

```bash
cp .env.test test/
```

## Test Structure

```
test/e2e/
├── README.md                           # This file
├── generic-test-flow.spec.ts           # Main S2S conversation E2E test suite
├── helpers/
│   ├── test.helper.ts                  # S2S conversation flow utilities
│   └── test-data.helper.ts             # Scenario scanning and config loading

test_data/
├── scenario_1/                         # Test scenario directory
│   ├── config.json                     # Profile and settings config
│   ├── 1_audio.wav                     # Audio files (numbered)
│   ├── 2_audio.wav
│   └── 3_audio.wav
└── scenario_2/
    ├── config.json
    ├── 1_audio.wav
    └── 2_audio.wav
```

### Test Data Format

#### config.json
```json
{
  "profile": "Nova Sonic v1 - Legacy",
  "voiceId": "matthew",
  "systemPrompt": "You are a helpful assistant.",
  "toolUse": { "tools": [] },
  "chatHistory": [],
  "speakFirst": false
}
```

**Configuration Options:**
- `profile` - Voice profile to use (e.g., "Nova Sonic v1 - Legacy")
- `voiceId` - Voice ID for responses (e.g., "matthew")
- `systemPrompt` - System instructions for the AI
- `toolUse` - Tool configuration (empty array if not needed)
- `chatHistory` - Initial chat history (empty array if not needed)
- `speakFirst` - If `true`, waits for bot message before playing user audio. If `false` (default), plays user audio immediately after recording initialization

## Running Tests

### All E2E Tests

```bash
npm run test:e2e
```

### Headed Mode (see browser)

```bash
npm run test:e2e:headed
```

### Debug Mode

```bash
npm run test:e2e:debug
```

### UI Mode (interactive)

```bash
npm run test:e2e:ui
```

### Specific Browser

```bash
npm run test:e2e:chromium
```

### Mobile Tests Only

```bash
npm run test:e2e:mobile
```

### View Test Report

```bash
npm run test:e2e:report
```

### Run Specific Test

```bash
npx playwright test --grep "Complete s2s test flow - Start to End"
```

## Test Scenarios

### 1. Complete S2S Conversation Flow - Start to End

**Description:** Full end-to-end S2S conversation flow from initialization to completion

**Steps:**
1. Navigate to S2S application
2. Select profile from dropdown (from `config.profile`)
3. Open config settings and configure:
   - Voice ID (`config.voiceId`)
   - System Prompt (`config.systemPrompt`)
   - Tool Use (`config.toolUse`)
   - Chat History (`config.chatHistory`)
   - ... 
4. Save settings
5. Start conversation (establish WebSocket connection)
6. Conduct multiple Q&A cycles (one per audio file):
   - Wait for 5s of silence, then trigger audio playback (user input from audio file)
   - Wait for audio playback to complete (user input)
   - Wait for bot audio response to finish playing (audio idle detection: 5+ seconds with no audio activity)
   - Add 2-second delay before next cycle
   - Repeat for each audio file (numbered: 1_*.wav, 2_*.wav, etc.)
7. Stop conversation

**Audio Cycle Control:**
The test uses audio-based cycle control rather than message counting:
- Each cycle waits for user response transcription using DOM selectors
- After user response, test waits for bot audio to go idle (no audio playback for 5+ seconds)
- Once idle, a 2-second delay is added before proceeding to the next cycle
- This ensures the bot's full response is heard before the next user input is played


## Troubleshooting

### WebSocket Connection Issues

**Problem:** Tests timeout waiting for WebSocket connection

**Solutions:**
- Verify Python WebSocket server is running on configured host/port
- Check `FRONTEND_URL` in `.env.test` is correct
- Verify firewall allows WebSocket connections
- Check browser console for connection errors

### Audio File Issues

**Problem:** "Audio file not found" or missing config

**Solutions:**
- Ensure test_data folder structure is correct (scenario folders with config.json and audio files)
- Verify audio files are numbered: `1_*.wav`, `2_*.wav`, etc.
- Check config.json is valid JSON with required fields
- Use default config if config.json is missing (values apply)

### Selector Not Found

**Problem:** Tests fail with "selector not found" or cannot find UI elements

**Solutions:**
- Run in headed mode to see actual UI: `npm run test:e2e:headed`
- Update selectors in `test.helper.ts` to match current S2S UI
- Add `data-testid` attributes to components for stable selectors
- Check if element is visible/enabled before interacting
- Use browser DevTools to inspect element selectors

### Missing Environment Variables

**Problem:** Tests reference undefined config values

**Solutions:**
- Copy `env.example` to `.env.test`
- Verify `ENABLE_AUDIO_TESTS=true` is set
- Set `FRONTEND_URL` to your S2S application URL


## Best Practices

1. **Use `data-testid` attributes** for stable selectors
2. **Keep tests independent** - each test should be self-contained
3. **Use helper functions** to reduce duplication
4. **Add meaningful assertions** - verify state changes
5. **Handle async operations** properly with `waitFor` methods
6. **Clean up after tests** - logout, clear state
7. **Use fixtures** for test data
8. **Document new tests** with clear descriptions