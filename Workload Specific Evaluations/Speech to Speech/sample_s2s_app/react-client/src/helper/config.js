// Model configuration with their capabilities
const ModelConfig = {
    "amazon.nova-sonic-v1:0": {
        name: "Amazon Nova Sonic v1",
        description: "no turn sensitivity, sentiment detection, or async tool calling",
        supportsTurnSensitivity: false,
        supportsParalinguisticDetection: false
    },
    "amazon.nova-2-sonic-v1:0": {
        name: "Amazon Nova Sonic v2",
        description: "Latest model version with turn sensitivity, async tool calling, and sentiment detection capabilities",
        supportsTurnSensitivity: true,
        supportsParalinguisticDetection: false
    }
};

const DemoProfiles = [
    {
        "name": "Nova Sonic v1 - Legacy",
        "description": "no turn sensitivity, sentiment detection, or async tool calling",
        "modelId": "amazon.nova-sonic-v1:0",
        "voiceId": "matthew",
        "systemPrompt": "You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios.",
        "inferenceConfig": {
            "maxTokens": 1024,
            "topP": 0.95,
            "temperature": 0.7
        },
        "toolConfig": {
            "tools": []
        }
    },
    {
        "name": "Nova Sonic v2 - New (Default)",
        "description": "Latest model version with turn sensitivity, async tool calling, and sentiment detection capabilities",
        "modelId": "amazon.nova-2-sonic-v1:0",
        "voiceId": "matthew",
        "systemPrompt": "You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios.",
        "inferenceConfig": {
            "maxTokens": 1024,
            "topP": 0.95,
            "temperature": 0.7
        },
        "toolConfig": {
            "tools": []
        }
    },
];

const VoicesByLanguage = {
    "English": {
        flag: "🇺🇸🇬🇧🇦🇺",
        voices: [
            {
                label: "Tiffany",
                value: "tiffany",
                locale: "en-US",
                accent: "US",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "English, French, Italian, German, and Spanish"
            },
            {
                label: "Matthew",
                value: "matthew",
                locale: "en-US",
                accent: "US",
                gender: "Male",
                polyglot: false
            },
            {
                label: "Amy",
                value: "amy",
                locale: "en-UK",
                accent: "UK",
                gender: "Female",
                polyglot: false
            },
            {
                label: "Olivia",
                value: "olivia",
                locale: "en-AU",
                accent: "AU",
                gender: "Female",
                polyglot: false
            }
        ]
    },
    "French": {
        flag: "🇫🇷",
        voices: [
            {
                label: "Florian",
                value: "florian",
                locale: "fr-FR",
                accent: "FR",
                gender: "Male",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Ambre",
                value: "ambre",
                locale: "fr-FR",
                accent: "FR",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            }
        ]
    },
    "Italian": {
        flag: "🇮🇹",
        voices: [
            {
                label: "Lorenzo",
                value: "lorenzo",
                locale: "it-IT",
                accent: "IT",
                gender: "Male",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Beatrice",
                value: "beatrice",
                locale: "it-IT",
                accent: "IT",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            }
        ]
    },
    "German": {
        flag: "🇩🇪",
        voices: [
            {
                label: "Lennart",
                value: "lennart",
                locale: "de-DE",
                accent: "DE",
                gender: "Male",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Tina",
                value: "tina",
                locale: "de-DE",
                accent: "DE",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Greta",
                value: "greta",
                locale: "de-DE",
                accent: "DE",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            }
        ]
    },
    "Spanish": {
        flag: "🇪🇸🇺🇸",
        voices: [
            {
                label: "Carlos",
                value: "carlos",
                locale: "en-US/es-US",
                accent: "US/ES",
                gender: "Male",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Lupe",
                value: "lupe",
                locale: "en-US/es-US",
                accent: "US/ES",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            }
        ]
    },
    "Portuguese": {
        flag: "🇧🇷",
        voices: [
            {
                label: "Camila",
                value: "camila",
                locale: "pt-BR",
                accent: "BR",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Leo",
                value: "leo",
                locale: "pt-BR",
                accent: "BR",
                gender: "Male",
                polyglot: true,
                polyglotLanguages: "with English"
            }
        ]
    },
    "Hindi": {
        flag: "🇮🇳",
        voices: [
            {
                label: "Aditi",
                value: "aditi",
                locale: "en-IN/hi-IN",
                accent: "IN",
                gender: "Female",
                polyglot: true,
                polyglotLanguages: "with English"
            },
            {
                label: "Rohan",
                value: "rohan",
                locale: "en-IN/hi-IN",
                accent: "IN",
                gender: "Male",
                polyglot: true,
                polyglotLanguages: "with English"
            }
        ]
    }
};

// Flatten voices for backward compatibility
const Voices = Object.values(VoicesByLanguage).flatMap(lang =>
    lang.voices.map(voice => ({
        label: `${voice.label} (${voice.accent})`,
        value: voice.value,
        ...voice
    }))
);

export { DemoProfiles, Voices, VoicesByLanguage, ModelConfig };