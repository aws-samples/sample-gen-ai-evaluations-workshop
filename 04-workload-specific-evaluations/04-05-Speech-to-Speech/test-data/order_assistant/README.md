# Pizza Order Assistant Test Scenario

## Overview

This scenario tests a conversational AI assistant for taking pizza orders at "Stefano's Pizza Palace." The assistant handles natural language ordering, menu inquiries, customization requests, and delivery arrangements through voice interaction.

## Purpose

- **Primary Goal:** Evaluate natural conversation flow for food ordering
- **Key Behaviors:** Menu navigation, order customization, upselling, order confirmation
- **Use Case:** Restaurant voice ordering system

## Scenario Details

### Profile Configuration
- **Voice Model:** Nova Sonic v1 - Legacy
- **Voice ID:** Matthew (friendly, approachable voice)
- **Conversation Style:** Casual, helpful customer service
- **Expected Duration:** 3-5 minutes (8 interaction turns)

### System Prompt Highlights

The assistant (Pizzabot) is configured to:
- Be friendly and efficient
- Help users place pizza orders through natural conversation
- Keep responses short and conversational
- Handle menu questions, customizations, and delivery details

## Audio Files

The scenario includes 8 pre-recorded audio files representing a complete ordering conversation:

- `01_ordering_pizza.wav` - Initial order request
- `02_what_specialties.wav` - Menu inquiry about specialty pizzas
- `03_bbq_chicken_ranch_interesting.wav` - Interest in specific pizza
- `04_get_extra_large.wav` - Size selection
- `05_thin_and_crispy.wav` - Crust preference
- `06_wings_flavors.wav` - Additional items inquiry
- `07_buffalo_wings.wav` - Wings selection
- `08_delivery.wav` - Delivery request

## Expected Conversation Flow

1. **Greeting:** Customer initiates order
2. **Menu Exploration:** Customer asks about specialty pizzas
3. **Selection:** Customer chooses BBQ Chicken Ranch pizza
4. **Customization:** Size (extra large) and crust (thin and crispy) selection
5. **Upselling:** Assistant suggests sides/drinks
6. **Additional Items:** Customer adds buffalo wings
7. **Delivery Details:** Customer requests delivery
8. **Confirmation:** Order summary and completion

## Success Criteria

### Speech Recognition
- Accurately transcribe food items (pizza, wings, BBQ chicken ranch, etc.)
- Handle casual speech patterns and food terminology

### Conversation Quality
- Natural, friendly tone throughout
- Appropriate responses to menu questions
- Smooth transitions between order stages
- Helpful suggestions without being pushy

### Order Management
- Track all order items correctly
- Remember customizations (size, crust type)
- Confirm order details accurately

### Customer Service
- Answer menu questions clearly
- Offer relevant suggestions
- Handle delivery request appropriately

## Evaluation Metrics

- **Speech Recognition Accuracy:** 95%+ for food terminology
- **Tool Calling Accuracy:** N/A (no tools configured for this scenario)
- **Response Relevance:** 100% appropriate to customer requests
- **Context Maintenance:** Remember all order items and preferences
- **Completeness:** Full order cycle from greeting to delivery
- **Clarity:** Clear, concise, friendly responses

## Difficulty Level

**Easy-Medium** - Straightforward ordering flow with clear stages, but requires maintaining order context and natural conversation.

## Sample Manual Test Conversation

If testing manually through the UI:

1. Click "Start Conversation"
2. Say: "Hi, I'd like to order a pizza for delivery"
3. When asked about preferences, say: "What specialty pizzas do you have?"
4. Choose a pizza: "I'll take the BBQ Chicken Ranch"
5. Specify size: "Make it extra large"
6. Add crust preference: "Thin and crispy crust, please"
7. When asked about sides: "Do you have wings? What flavors?"
8. Add wings: "I'll get the buffalo wings"
9. Confirm delivery: "Yes, delivery please"

## Notes

- This scenario tests basic conversational ordering without complex tool integrations
- Focus is on natural language understanding and order tracking
- Assistant should maintain friendly, efficient tone throughout
- No payment processing or address collection in this simplified scenario
