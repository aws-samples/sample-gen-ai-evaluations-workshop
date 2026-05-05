# Test Scenario Configuration Template

This document describes the structure and metadata fields used in test scenario configurations.

## Directory Structure

Each test scenario should be organized as follows:

```
test-data/
└── scenario_name/
    ├── README.md              # Detailed scenario documentation
    ├── config.json            # Configuration with metadata
    ├── 01_audio_file.wav      # Numbered audio files
    ├── 02_audio_file.wav
    └── ...
```

## config.json Structure

### Required Metadata Fields

```json
{
  "description": "Brief description of what this scenario tests",
  "expected_turns": 10,
  "key_behaviors": ["behavior1", "behavior2", "behavior3"],
  "difficulty_level": "Easy|Easy-Medium|Medium|Medium-High|High",
  
  "profile": "Nova Sonic v1 - Legacy",
  "voiceId": "matthew",
  "systemPrompt": "...",
  "toolUse": {"tools": []},
  "chatHistory": [],
  "saveAudio": false,
  "inferenceConfig": {
    "maxTokens": 1024,
    "temperature": 0.7,
    "topP": 0.95
  }
}
```

### Metadata Field Descriptions

- **description**: One-sentence summary of the scenario's purpose and what it tests
- **expected_turns**: Approximate number of conversation turns (user + assistant exchanges)
- **key_behaviors**: Array of 3-5 key behaviors being evaluated (e.g., "context maintenance", "tool calling")
- **difficulty_level**: Complexity rating from Easy to High

### Standard Configuration Fields

- **profile**: Nova Sonic model version to use
- **voiceId**: Voice identifier for responses (e.g., "matthew", "joanna")
- **systemPrompt**: Complete system prompt defining assistant behavior
- **toolUse**: Tool configuration (empty array if no tools)
- **chatHistory**: Initial conversation history (usually empty array)
- **saveAudio**: Whether to record audio interactions
- **inferenceConfig**: Model inference parameters (tokens, temperature, topP)

## README.md Template

Each scenario should include a README.md with these sections:

1. **Overview** - Brief description and purpose
2. **Purpose** - Primary goal, key behaviors, use case
3. **Scenario Details** - Profile configuration and system prompt highlights
4. **Audio Files** - List and description of each audio file
5. **Expected Conversation Flow** - Step-by-step flow description
6. **Success Criteria** - What defines a successful evaluation
7. **Evaluation Metrics** - Specific metrics and target values
8. **Difficulty Level** - Complexity rating with explanation
9. **Sample Manual Test Conversation** - Instructions for manual testing
10. **Notes** - Additional important information

## Audio File Naming Convention

Audio files should be numbered sequentially:

- `01_descriptive_name.wav`
- `02_descriptive_name.wav`
- `03_descriptive_name.wav`

The numbering determines playback order during automated tests.

## Creating New Scenarios

1. Create a new directory under `test-data/`
2. Add numbered audio files (.wav format)
3. Create config.json with all required fields
4. Write comprehensive README.md
5. Add validation data to `data/s2s_validation_dataset.jsonl`
6. Test the scenario using Playwright tests

## Example Scenarios

See existing scenarios for reference:
- `test-data/interview_practice_assistant/` - Complex interview flow
- `test-data/order_assistant/` - Simple ordering conversation
- `test-data/survey_assistant/` - Structured survey questions

